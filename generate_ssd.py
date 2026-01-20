import argparse
import torch
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def top_k_logits(logits, k):
    if k <= 0:
        return logits
    else:
        values, _ = torch.topk(logits, k)
        min_values = values[..., -1, None]
        return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)


def top_p_logits(logits, p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_mask = cumulative_probs > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    mask_indices = torch.scatter(torch.full_like(logits, False, dtype=torch.bool),
                                 -1, sorted_indices, sorted_mask)
    logits = logits.masked_fill(mask_indices, float('-inf'))
    return logits


def sample_with_temperature_topk_topp(logits, temperature=1.0, top_k=0, top_p=1.0):
    orig_shape = logits.shape[:-1]    # [batch, block]
    vocab_size = logits.shape[-1]

    logits = logits.reshape(-1, vocab_size)  # [batch*block, vocab]

    if temperature != 1.0:
        logits = logits / temperature
    if top_k > 0:
        logits = top_k_logits(logits, top_k)
    if top_p < 1.0:
        logits = top_p_logits(logits, top_p)
    probs = F.softmax(logits, dim=-1)  # shape: [batch*block, vocab]
    assert probs.dim() == 2
    token = torch.multinomial(probs, num_samples=1)  # [batch*block, 1]
    token_prob = torch.gather(probs, -1, token)     # [batch*block, 1]

    return token.view(*orig_shape), token_prob.view(*orig_shape)


def get_num_transfer_tokens(block_length, steps):
    base = block_length // steps
    remainder = block_length % steps
    num_transfer_tokens = torch.zeros(steps, dtype=torch.int64) + base
    num_transfer_tokens[:remainder] += 1
    return num_transfer_tokens


def _probs_from_logits(logits_2d: torch.Tensor, *, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    """
    Convert logits -> proposal/target probs using the same sampling transforms as draft sampling.

    logits_2d: (N, V)
    returns: (N, V) float probs
    """
    if temperature != 1.0:
        logits_2d = logits_2d / temperature
    if top_k and top_k > 0:
        logits_2d = top_k_logits(logits_2d, top_k)
    if top_p is not None and top_p < 1.0:
        logits_2d = top_p_logits(logits_2d, top_p)
    return F.softmax(logits_2d, dim=-1)


def _reject_resample_from_delta(q_probs_1d: torch.Tensor, p_probs_1d: torch.Tensor) -> torch.Tensor:
    """
    Residual resampling used in SSD: sample from normalized (q - p)+.
    Falls back to sampling from q if the residual mass is ~0.
    """
    delta = (q_probs_1d - p_probs_1d).clamp_min(0.0)
    z = float(delta.sum().item())
    if z <= 0.0 or not torch.isfinite(delta).all():
        # Fallback: sample from q directly.
        return torch.multinomial(q_probs_1d.clamp_min(0).to(dtype=torch.float32), num_samples=1).squeeze(0)
    delta = (delta / z).to(dtype=torch.float32)
    return torch.multinomial(delta, num_samples=1).squeeze(0)


def _construct_2l_verifier_attention_mask_bool(*, L: int, cache_len: int, device: torch.device) -> torch.Tensor:
    """
    2L trick attention mask for SDAR inference path (bool mask where True means "allowed").

    Keys are laid out as: [cached keys (cache_len) | first-half (L) | second-half masks (L)]
    Queries are: [first-half (L) | second-half masks (L)]
    """
    # (2L, cache_len + 2L)
    attn = torch.zeros((2 * L, cache_len + 2 * L), device=device, dtype=torch.bool)
    if cache_len > 0:
        attn[:, :cache_len] = True

    # First half: token-causal within first-half.
    attn[:L, cache_len:cache_len + L] = torch.tril(torch.ones((L, L), device=device, dtype=torch.bool), diagonal=0)

    # Second half: strict-causal into first-half (exclude same-position draft token).
    attn[L:, cache_len:cache_len + L] = torch.tril(torch.ones((L, L), device=device, dtype=torch.bool), diagonal=-1)

    # Second half: block-diagonal self-attention (block_size=1 => identity).
    attn[L:, cache_len + L:cache_len + 2 * L] = torch.eye(L, device=device, dtype=torch.bool)
    return attn.unsqueeze(0)


def _find_mask_spans_1d(mask_1d: torch.Tensor) -> list[list[int]]:
    """
    Find all contiguous spans of True in a 1D boolean tensor.
    Returns a list of spans, each span is a list of indices (L2R).
    """
    assert mask_1d.dim() == 1
    spans: list[list[int]] = []
    i = 0
    L = int(mask_1d.shape[0])
    while i < L:
        if not bool(mask_1d[i].item()):
            i += 1
            continue
        j = i
        while j < L and bool(mask_1d[j].item()):
            j += 1
        spans.append(list(range(i, j)))
        i = j
    return spans


@torch.no_grad()
def block_diffusion_generate(
        model,
        prompt,
        mask_id,
        gen_length=128,
        block_length=8,
        denoising_steps=8,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        remasking_strategy='low_confidence_dynamic',
        confidence_threshold=0.85,
        eb_threshold=None,
        stopping_criteria_idx=None
    ):

    model.eval()
    input_ids = prompt['input_ids']
    prompt_length = input_ids.shape[1]
    past_key_values = DynamicCache()

    num_blocks = (prompt_length + gen_length +
                  block_length - 1) // block_length
    total_length = num_blocks * block_length

    block_mask = torch.tril(torch.ones(
        num_blocks, num_blocks, device=model.device))
    block_diffusion_attention_mask = block_mask.repeat_interleave(block_length, dim=0)\
                                               .repeat_interleave(block_length, dim=1).unsqueeze(0)
    verifier_attention_mask = torch.tril(torch.ones(total_length, total_length, device=model.device)).unsqueeze(0)
    position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)

    x = torch.full((1, total_length), mask_id,
                   dtype=torch.long, device=model.device)
    x[:, :prompt_length] = input_ids
    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length

    # Prefill stage
    if prefill_length > 0:
        cur_x = x[:, :prefill_length]
        cur_attn_mask = block_diffusion_attention_mask[:,
                                                       :prefill_length, :prefill_length]
        cur_position_ids = position_ids[:, :prefill_length]
        model(cur_x,
              attention_mask=cur_attn_mask,
              position_ids=cur_position_ids,
              past_key_values=past_key_values,
              use_cache=True,
              store_kv=True)

    num_transfer_tokens = get_num_transfer_tokens(
        block_length, denoising_steps)

    # Decode stage
    for num_block in range(prefill_blocks, num_blocks):
        cur_x = x[:, num_block*block_length:(num_block+1)*block_length].clone()
        cur_attn_mask = block_diffusion_attention_mask[
            :, num_block*block_length:(num_block+1)*block_length, :(num_block+1)*block_length
        ]
        cur_ver_attn_mask = verifier_attention_mask[
            :, num_block*block_length:(num_block+1)*block_length, :(num_block+1)*block_length
        ]
        cur_position_ids = position_ids[:, num_block *
                                        block_length:(num_block+1)*block_length]
        for step in range(denoising_steps + 1):
            mask_index = (cur_x == mask_id)
            if mask_index.sum() == 0:
                # Store kv cache
                model(
                    cur_x,
                    attention_mask=cur_attn_mask,
                    # attention_mask=cur_ver_attn_mask,
                    position_ids=cur_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    store_kv=True)
                break

            # Denosing
            # first_mask_index_rel = _find_mask_spans_1d(mask_index[0])[0][0]
            # if first_mask_index_rel > 0:
            #     cur_attn_mask[:, :first_mask_index_rel, :] = cur_ver_attn_mask[:, :first_mask_index_rel, :]
            logits = model(
                cur_x,
                attention_mask=cur_ver_attn_mask,
                position_ids=cur_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                store_kv=False).logits

            # Sampling
            x0, x0_p = sample_with_temperature_topk_topp(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

            # Self-speculative decoding (SSD) with block_size=1 verifier (static: one block at a time).
            # - draft: current block diffusion sampling (x0, x0_p)
            # - verify: 2L trick + rejection sampling over first contiguous mask span
            # - fallback: if SSD decodes < num_transfer_tokens[step], use original remasking_strategy
            target_k = int(num_transfer_tokens[min(step, len(num_transfer_tokens) - 1)].item())
            decoded_this_step = 0

            # Draft-fill the entire current block for verifier's first half.
            input_seq_ver_first = cur_x.clone()
            if mask_index.any():
                input_seq_ver_first[mask_index] = x0[mask_index]

            # Find first contiguous mask span within this block (L2R), capped at target_k.
            spans_rel = _find_mask_spans_1d(mask_index[0])
            span_rel: list[int] = spans_rel[0] if len(spans_rel) > 0 else []

            resampled_rel: list[int] = []
            if len(span_rel) > 0:
                L = int(cur_x.shape[1])
                cache_len = int(past_key_values.get_seq_length()) if past_key_values is not None else 0

                # 2L verifier input: [draft-filled | all-mask], take logits from masked half.
                mask_tokens = torch.full_like(input_seq_ver_first, mask_id)
                input_seq_ver = torch.cat([input_seq_ver_first, mask_tokens], dim=1)  # (1, 2L)

                verify_attn_mask = _construct_2l_verifier_attention_mask_bool(
                    L=L, cache_len=cache_len, device=cur_x.device
                )  # (1, 2L, cache_len+2L)
                verify_position_ids = torch.cat([cur_position_ids, cur_position_ids], dim=1)  # (1, 2L)

                verify_logits = model(
                    input_seq_ver,
                    attention_mask=verify_attn_mask,
                    position_ids=verify_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    store_kv=False,
                ).logits

                span_rel_t = torch.tensor(span_rel, device=cur_x.device, dtype=torch.long)
                q_logits = verify_logits[0, span_rel_t + L, :]  # (S, V) from masked half
                q_probs = _probs_from_logits(q_logits, temperature=temperature, top_k=top_k, top_p=top_p)  # (S, V)

                draft_tokens = x0[0, span_rel_t]  # (S,)
                p_sel = x0_p[0, span_rel_t].clamp_min(1e-12)  # (S,)
                q_sel = q_probs.gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1).clamp_min(0.0)  # (S,)

                ratios = (q_sel / p_sel).clamp(max=1.0)
                accept_flags = torch.rand((ratios.shape[0],), device=ratios.device, dtype=ratios.dtype) < ratios
                reject_idx = (~accept_flags).nonzero(as_tuple=True)[0]
                first_reject = int(reject_idx[0].item()) if reject_idx.numel() > 0 else None

                # Build an ordered update list (L2R): accepted prefix, and optionally the first rejected position.
                update_rel: list[int] = []
                if first_reject is None:
                    # All accepted.
                    update_rel = span_rel
                else:
                    # Accept prefix.
                    if first_reject > 0:
                        update_rel.extend(span_rel[:first_reject])

                    # At first rejection, residual-resample and always unmask it.
                    rej_rel = int(span_rel[first_reject])
                    update_rel.append(rej_rel)

                # NOTE: `target_k` here is treated as a *minimum* number of tokens to unmask per step.
                # SSD is allowed to accept more than `target_k` (e.g. the whole first span); the fallback
                # only kicks in when SSD unmasked fewer than `target_k`.

                if len(update_rel) > 0:
                    upd_rel_t = torch.tensor(update_rel, device=cur_x.device, dtype=torch.long)
                    # For accepted positions, apply draft tokens. For the first rejected position (if included),
                    # apply residual-resampled token.
                    if first_reject is None:
                        cur_x[0, upd_rel_t] = x0[0, upd_rel_t]
                    else:
                        # Accepted prefix (if any)
                        if first_reject > 0:
                            acc_rel = [r for r in update_rel if r in span_rel[:first_reject]]
                            if len(acc_rel) > 0:
                                acc_rel_t = torch.tensor(acc_rel, device=cur_x.device, dtype=torch.long)
                                cur_x[0, acc_rel_t] = x0[0, acc_rel_t]

                        # Rejection token (only if it's within committed set)
                        if rej_rel in update_rel:
                            rej_q = q_probs[first_reject]  # (V,)
                            rej_p = _probs_from_logits(
                                logits[0, rej_rel, :].unsqueeze(0),
                                temperature=temperature,
                                top_k=top_k,
                                top_p=top_p,
                            )[0]
                            resampled = _reject_resample_from_delta(rej_q, rej_p)
                            cur_x[0, rej_rel] = resampled.to(dtype=torch.long)
                            resampled_rel.append(rej_rel)

                    decoded_this_step += len(update_rel)

            # If SSD didn't decode enough, use existing remasking strategy for remaining masked positions.
            if decoded_this_step < target_k:
                k_left = target_k - decoded_this_step
                mask_index = (cur_x == mask_id)  # update after SSD

            # Sampling strategy
            if remasking_strategy == 'sequential':
                if decoded_this_step < target_k:
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                    for j in range(cur_x.shape[0]):
                        if not mask_index[j].any():
                            continue
                        first_mask_index = int(mask_index[j].nonzero(as_tuple=True)[0].min().item())
                        picked = 0
                        idx = first_mask_index
                        while idx < cur_x.shape[1] and picked < k_left:
                            if bool(mask_index[j, idx].item()):
                                transfer_index[j, idx] = True
                                picked += 1
                            idx += 1
                else:
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool)

            elif remasking_strategy == 'low_confidence_static':
                if decoded_this_step < target_k:
                    confidence = torch.where(mask_index, x0_p, -torch.inf)
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                    for j in range(confidence.shape[0]):
                        k = min(int(k_left), int(mask_index[j].sum().item()))
                        if k <= 0:
                            continue
                        _, idx = torch.topk(confidence[j], k)
                        transfer_index[j, idx] = True
                else:
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool)

            elif remasking_strategy == 'low_confidence_dynamic':
                if decoded_this_step < target_k:
                    confidence = torch.where(mask_index, x0_p, -torch.inf)
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                    for j in range(confidence.shape[0]):
                        k = min(int(k_left), int(mask_index[j].sum().item()))
                        if k <= 0:
                            continue
                        high_conf_mask = confidence[j] > confidence_threshold
                        num_high_confidence = int(high_conf_mask.sum().item())
                        if num_high_confidence >= k:
                            # If too many pass the threshold, keep the highest-confidence k.
                            vals = torch.where(high_conf_mask, confidence[j], -torch.inf)
                            _, idx = torch.topk(vals, k)
                            transfer_index[j, idx] = True
                        else:
                            _, idx = torch.topk(confidence[j], k)
                            transfer_index[j, idx] = True
                else:
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            elif remasking_strategy == "entropy_bounded":
                if decoded_this_step < target_k:
                    eps = 1e-12
                    entropies = -(x0_p.clamp_min(eps) * (x0_p.clamp_min(eps)).log()).sum(dim=-1)
                    entropies = torch.where(mask_index, entropies, torch.inf)
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                    ent_sorted, order = torch.sort(entropies, dim=1, descending=False)
                    cumsum = torch.cumsum(ent_sorted, dim=1)
                    for j in range(x0_p.shape[0]):
                        total_masked = int(mask_index[j].sum().item())
                        if total_masked <= 0:
                            continue
                        k = int(torch.searchsorted(cumsum[j], torch.tensor(eb_threshold, device=x0_p.device), right=False).item())
                        k = max(1, min(k, total_masked, int(k_left)))
                        selected_token_indices = order[j, :k]
                        transfer_index[j, selected_token_indices] = True
                else:
                    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                
            else:
                raise ValueError(
                    f"Unknown remasking strategy: {remasking_strategy}")

            cur_x[transfer_index] = x0[transfer_index]

        x[:, num_block*block_length:(num_block+1)*block_length] = cur_x
        if stopping_criteria_idx is not None and any(stop_idx in x[:, prompt_length:] for stop_idx in stopping_criteria_idx):
            break

    return x


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the pretrained model directory")
    parser.add_argument("--trust_remote_code", action='store_true')
    parser.add_argument("--mask_id", type=int, default=None,
                        help="Mask token id for Diffusion")
    parser.add_argument("--prompt_length", type=int, default=4096,
                        help="Maximum prompt length in tokens")
    parser.add_argument("--gen_length", type=int, default=2048,
                        help="Maximum generation length in tokens")
    parser.add_argument("--block_length", type=int, default=4,
                        help="Length of token block to replace each denoising step")
    parser.add_argument("--denoising_steps", type=int, default=4,
                        help="Number of denoising steps (iterations)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Top-K sampling (0 to disable)")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-P sampling probability threshold")
    parser.add_argument("--remasking_strategy", type=str, default="low_confidence_dynamic",
                        choices=["low_confidence_dynamic",
                                 "low_confidence_static",
                                 "sequential",
                                 "entropy_bounded"],
                        help="Strategy for remasking tokens")
    parser.add_argument("--confidence_threshold", type=float, default=0.85,
                        help="Confidence threshold for low-confidence remasking")
    parser.add_argument("--eb_threshold", type=float, default=0.35,
                        help="entropy threshold for entropy bounded sampling")
    parser.add_argument("--stopping_criteria_idx", type=int, nargs="+", default=None,
                        help="List of token IDs that stop generation (e.g. eos_token_id)")

    parser.add_argument("--device", type=str, default="cuda",)
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16"],)
    return parser.parse_args(), parser


if __name__ == "__main__":
    args, parser = parse_args()

    if args.remasking_strategy == "low_confidence_dynamic" and args.confidence_threshold is None:
        parser.error(
            "--confidence_threshold is required when --remasking_strategy=low_confidence_dynamic"
        )
    if args.remasking_strategy == "entropy_bounded" and args.eb_threshold is None:
        parser.error(
            "--eb_threshold is required when --remasking_strategy=entropy_bounded"
        )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=args.dtype,
        device_map=args.device
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=args.trust_remote_code,
    )

    if args.mask_id is None:
        args.mask_id = tokenizer(tokenizer.mask_token)['input_ids'][0]
    if args.stopping_criteria_idx is None:
        gen_cfg = GenerationConfig.from_pretrained(args.model_dir,)
        args.stopping_criteria_idx = gen_cfg.eos_token_id
    if isinstance(args.stopping_criteria_idx, int):
        args.stopping_criteria_idx = [args.stopping_criteria_idx,]
    args.stop_words = tokenizer.convert_ids_to_tokens(
        args.stopping_criteria_idx)
    print(f"Your Arguments: {args}")

    origin_prompt = [
        # dict(role="user", content="Given the function $f(x) = \\frac{4x^2 - 4x + 4}{x^2 + 2x + 4}$, where $x \\in \\mathbb{R}$, determine its minimum value.\nPlease reason step by step, and put your final answer within \\boxed{}.\n"),
        # dict(role="user", content="If the domain of the function $\\log x^2$ is $x < a$ or $x > b$, for some $a$ and $b$, find $a + b$.\nPlease reason step by step, and put your final answer within \\boxed{}.\n")
        dict(role="user", content="Question: John takes care of 10 dogs. Each dog takes 0.5 hours a day to walk and take care of their business. How many hours a week does he spend taking care of dogs?\nPlease reason step by step, and put your final answer within \\boxed{}.\n")
    ]

    messages = tokenizer.apply_chat_template(
        origin_prompt, add_generation_prompt=True, tokenize=False)
    tokenize_kwargs = dict(
        return_tensors='pt',
        padding=True,
        truncation=True,
        add_special_tokens=False,
        max_length=args.prompt_length
    )

    tokens = tokenizer.batch_encode_plus([messages], **tokenize_kwargs)
    tokens = {k: v.to(model.device) for k, v in tokens.items()}

    output_ids = block_diffusion_generate(
        model,
        prompt=tokens,
        mask_id=args.mask_id,
        gen_length=args.gen_length,
        block_length=args.block_length,
        denoising_steps=args.denoising_steps,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        remasking_strategy=args.remasking_strategy,
        confidence_threshold=args.confidence_threshold,
        eb_threshold=args.eb_threshold,
        stopping_criteria_idx=args.stopping_criteria_idx
    )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    cleaned_text = output_text.replace('<|MASK|>', '')
    print(cleaned_text)
