#!/usr/bin/env python3
"""
Evaluate an SDAR (block diffusion) model on GSM8K.

This mirrors the workflow/outputs of `reference/eval_gsm8k_llada.py`, but uses
`block_diffusion_generate()` from `generate.py` for inference.
"""

import warnings

warnings.filterwarnings("ignore")

import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# from generate import block_diffusion_generate  # noqa: E402

SYSTEM_PROMPT = "Solve the following math problem concisely and clearly and put your final answer within \\boxed{}."

SYSTEM_PROMPT_COT = (
    "Solve the following math problem step by step. "
    "Show your reasoning clearly. "
    "Put your final answer within \\boxed{}."
)

FEW_SHOT_PROMPT = """Here are a few examples of how to solve math problems step by step.

Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: There are 15 trees originally. After planting, there are 21 trees. The number of trees planted is 21 - 15 = 6. So the final answer is \\boxed{6}.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Leah has 32 chocolates and her sister has 42, so together they have 32 + 42 = 74 chocolates. They eat 35 of them, leaving 74 - 35 = 39 chocolates. So the final answer is \\boxed{39}.

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer: Jason starts with 20 lollipops and ends with 12, so he gave away 20 - 12 = 8 lollipops. So the final answer is \\boxed{8}.

Now solve a new problem in the same style.

Question: {question}
Answer:"""


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ascii_sparkline(values, max_len=50):
    if values is None or len(values) == 0:
        return ""
    n = len(values)
    if n > max_len and max_len > 0:
        step = n / float(max_len)
        idxs = [min(int(i * step), n - 1) for i in range(max_len)]
        vals = [values[i] for i in idxs]
    else:
        vals = list(values)
    vmin = min(vals)
    vmax = max(vals)
    if math.isclose(vmax, vmin):
        norm = [0.0 for _ in vals]
    else:
        rng = vmax - vmin
        norm = [(v - vmin) / rng for v in vals]
    blocks = "▁▂▃▄▅▆▇█"
    try:
        chars = [blocks[min(int(x * (len(blocks) - 1)), len(blocks) - 1)] for x in norm]
        return "".join(chars)
    except Exception:
        shades = " .:-=+*#%@"
        chars = [shades[min(int(x * (len(shades) - 1)), len(shades) - 1)] for x in norm]
        return "".join(chars)


def extract_ground_truth(ans_text: str):
    """Extract numeric ground-truth answer from GSM8K annotation."""
    if not isinstance(ans_text, str):
        return None
    m = re.findall(r"####\s*([-+]?\d+(?:\.\d+)?)", ans_text)
    if m:
        try:
            return float(m[-1]) if "." in m[-1] else int(m[-1])
        except Exception:
            pass
    nums = re.findall(r"([-+]?\d+(?:\.\d+)?)", ans_text)
    if nums:
        try:
            return float(nums[-1]) if "." in nums[-1] else int(nums[-1])
        except Exception:
            return None
    return None


def extract_predicted_answer(text: str):
    """Extract final numeric answer from model output."""
    if not isinstance(text, str):
        return None
    boxed = re.findall(r"\\boxed\{\s*([-+]?\d+(?:\.\d+)?)\s*\}", text)
    if boxed:
        try:
            return float(boxed[-1]) if "." in boxed[-1] else int(boxed[-1])
        except Exception:
            pass
    m = re.findall(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    if m:
        try:
            return float(m[-1]) if "." in m[-1] else int(m[-1])
        except Exception:
            pass
    nums = re.findall(r"([-+]?\d+(?:\.\d+)?)", text)
    if nums:
        try:
            return float(nums[-1]) if "." in nums[-1] else int(nums[-1])
        except Exception:
            return None
    return None


def build_user_prompt(question: str, use_few_shot: bool, cot: bool) -> str:
    if use_few_shot:
        return FEW_SHOT_PROMPT.replace("{question}", question)
    if cot:
        return f"Question: {question}\nAnswer: Let's think step by step."
    # return f"Question: {question}\nAnswer:"
    return f"Question: {question}"


def _normalize_stop_ids(tokenizer, stopping_criteria_idx):
    if stopping_criteria_idx is None:
        return None
    if isinstance(stopping_criteria_idx, int):
        return [stopping_criteria_idx]
    return list(stopping_criteria_idx)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GSM8K with an SDAR block-diffusion model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model_dir", default="JetLM/SDAR-8B-Chat", type=str, help="Path to pretrained model directory")
    # model_group.add_argument("--trust_remote_code", action="store_true")
    model_group.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for model weights",
    )
    model_group.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help='Device map for HF loading (e.g. "auto").',
    )
    model_group.add_argument(
        "--cuda-visible-devices",
        "--cvd",
        type=str,
        default=None,
        help="Set CUDA_VISIBLE_DEVICES before loading the model",
    )
    model_group.add_argument("--mask_id", type=int, default=None, help="Mask token id for diffusion")

    # Generation settings (SDAR)
    gen_group = parser.add_argument_group("Generation Settings (SDAR)")
    gen_group.add_argument("--prompt_length", type=int, default=4096, help="Maximum prompt length in tokens")
    gen_group.add_argument("--gen_length", type=int, default=256, help="Maximum generation length in tokens")
    gen_group.add_argument("--block_length", type=int, default=4, help="Block length for diffusion decoding")
    gen_group.add_argument("--denoising_steps", type=int, default=4, help="Number of denoising steps per block")
    gen_group.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (>0)")
    gen_group.add_argument("--top_k", type=int, default=0, help="Top-K sampling (0 disables)")
    gen_group.add_argument("--top_p", type=float, default=1.0, help="Top-P sampling probability threshold")
    gen_group.add_argument(
        "--remasking_strategy",
        type=str,
        default="low_confidence_dynamic",
        choices=["low_confidence_dynamic", "low_confidence_static", "sequential", "entropy_bounded"],
        help="Strategy for remasking tokens",
    )
    gen_group.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.85,
        help="Confidence threshold for low-confidence dynamic remasking",
    )
    gen_group.add_argument(
        "--eb_threshold",
        type=float,
        default=0.35,
        help="Entropy threshold for entropy bounded sampling",
    )
    gen_group.add_argument(
        "--stopping_criteria_idx",
        type=int,
        nargs="+",
        default=None,
        help="List of token IDs that stop generation (e.g. eos_token_id)",
    )
    gen_group.add_argument("--use_ssd", action="store_true", help="Use SSD (Self-Speculative Decoding)")

    # Evaluation settings
    eval_group = parser.add_argument_group("Evaluation Settings")
    eval_group.add_argument("--sample_n", type=int, default=10, help="Sample N test examples deterministically")
    eval_group.add_argument("--few_shot", action="store_true", help="Use 3-shot prompt examples")
    eval_group.add_argument("--cot", action="store_true", help="Enable chain-of-thought prompting")
    eval_group.add_argument("--no_chat", action="store_true", help="Do not use tokenizer chat template")
    eval_group.add_argument("--use_majority_vote", action="store_true", help="Majority vote over multiple samples")
    eval_group.add_argument("--n_votes", type=int, default=5, help="Number of votes when majority voting")
    eval_group.add_argument("--verbose", action="store_true", help="Print per-example generations and predictions")

    # Logging
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument("--log_samples", action="store_true", help="Save per-sample outputs and summary")
    log_group.add_argument("--output_path", type=str, default=None, help="Path to write results (json/jsonl)")

    # Other settings
    other_group = parser.add_argument_group("Other Settings")
    other_group.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    if args.gen_length <= 0:
        parser.error("--gen_length must be positive")
    if args.block_length <= 0:
        parser.error("--block_length must be positive")
    if args.denoising_steps < 0:
        parser.error("--denoising_steps must be non-negative")
    if args.remasking_strategy == "low_confidence_dynamic" and args.confidence_threshold is None:
        parser.error("--confidence_threshold is required when --remasking_strategy=low_confidence_dynamic")
    if args.remasking_strategy == "entropy_bounded" and args.eb_threshold is None:
        parser.error("--eb_threshold is required when --remasking_strategy=entropy_bounded")

    set_seed(args.seed)

    print("\nConfiguration:")
    print(f"  Model: {args.model_dir}")
    print(f"  Use SSD: {args.use_ssd}")
    print(f"  Dtype: {args.dtype}")
    print(f"  Device map: {args.device_map}")
    print(f"  Mask id: {args.mask_id}")
    print(f"  Prompt length: {args.prompt_length}")
    print(f"  Gen length: {args.gen_length}")
    print(f"  Block length: {args.block_length}")
    print(f"  Denoising steps: {args.denoising_steps}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-k: {args.top_k}")
    print(f"  Top-p: {args.top_p}")
    print(f"  Remasking: {args.remasking_strategy}")
    print(f"  Few-shot: {args.few_shot}")
    print(f"  CoT: {args.cot}")
    print(f"  Majority vote: {args.use_majority_vote} (n={args.n_votes})")
    print(f"  Seed: {args.seed}")

    if args.use_ssd:
        from generate_ssd import block_diffusion_generate
    else:
        from generate import block_diffusion_generate
    generate_func = block_diffusion_generate

    print("\nLoading model and tokenizer...")
    if args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    if args.mask_id is None:
        if tokenizer.mask_token is None:
            parser.error("--mask_id is required because tokenizer.mask_token is None")
        args.mask_id = tokenizer(tokenizer.mask_token)["input_ids"][0]

    if args.stopping_criteria_idx is None:
        try:
            gen_cfg = GenerationConfig.from_pretrained(args.model_dir)
            args.stopping_criteria_idx = gen_cfg.eos_token_id
        except Exception:
            args.stopping_criteria_idx = None
    args.stopping_criteria_idx = _normalize_stop_ids(tokenizer, args.stopping_criteria_idx)

    print("\nLoading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="test")

    if args.sample_n is not None and args.sample_n > 0 and args.sample_n < len(dataset):
        rng = random.Random(123456)  # fixed for repeatability across runs
        indices = list(range(len(dataset)))
        rng.shuffle(indices)
        select = indices[: args.sample_n]
        select.sort()
        print(f"Selected indices (n={len(select)}): {select}")
        dataset = dataset.select(select)

    datasize = len(dataset)
    print("gsm8k test size:", datasize)

    results = []
    correct_so_far = 0
    acc_series = []

    mask_token_str = tokenizer.mask_token or "<|MASK|>"
    stop_ids = args.stopping_criteria_idx

    pbar = tqdm(range(datasize), desc="Evaluating")
    for idx in pbar:
        ex = dataset[idx]
        question = ex["question"]
        gold_text = ex["answer"]
        gt = extract_ground_truth(gold_text)

        user_content = build_user_prompt(question, use_few_shot=args.few_shot, cot=args.cot)
        system_prompt = SYSTEM_PROMPT_COT if args.cot else SYSTEM_PROMPT
        prompt_text = f"{system_prompt}\n\n{user_content}"

        if args.no_chat:
            formatted = prompt_text
        else:
            messages = [{"role": "user", "content": prompt_text}]
            formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        tokenize_kwargs = dict(
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
            max_length=args.prompt_length,
        )
        tokens = tokenizer.batch_encode_plus([formatted], **tokenize_kwargs)
        tokens = {k: v.to(model.device) for k, v in tokens.items()}
        prompt_len = int(tokens["input_ids"].shape[1])

        batch_model_answers = []
        n_votes = args.n_votes if args.use_majority_vote else 1

        for vote_i in range(n_votes):
            # Make stochastic sampling vary across votes while remaining reproducible.
            if n_votes > 1:
                set_seed(args.seed + 1000 + vote_i)

            out_ids = generate_func(
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
                stopping_criteria_idx=stop_ids,
            )

            gen_ids = out_ids[:, prompt_len : prompt_len + args.gen_length]
            out_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
            out_text = out_text.replace(mask_token_str, "").replace("<|MASK|>", "")
            pred_num = extract_predicted_answer(out_text)
            batch_model_answers.append({"text": out_text, "numeric": pred_num})

        numeric_answers = [ma["numeric"] for ma in batch_model_answers]
        filtered = [num for num in numeric_answers if num is not None]
        majority = Counter(filtered).most_common(1)[0][0] if filtered else None

        correct = (majority == gt) if majority is not None and gt is not None else False
        if correct:
            correct_so_far += 1

        results.append(
            {
                "question": question,
                "gold_answer_text": gold_text,
                "model_answers_text": [ma["text"] for ma in batch_model_answers],
                "extracted_model_answers": numeric_answers,
                "extracted_gold_answer": gt,
                "majority_answer": majority,
                "correct": correct,
            }
        )

        if args.verbose:
            print(f"\n[Example {idx}]")
            print(f"Question: {question}")
            print(f"Gold answer text: {gold_text}")
            print(f"Model answer text: {batch_model_answers[0]['text']}")
            print(f"Predictions: {numeric_answers}")
            print(f"Majority: {majority}, GT: {gt}, Correct: {correct}")

        processed = len(results)
        acc_so_far = correct_so_far / float(processed)
        acc_series.append(acc_so_far)
        pbar.set_postfix({"acc": f"{acc_so_far:.4f}", "trend": ascii_sparkline(acc_series, max_len=80)})

    cnt = sum(1 for r in results if r["correct"])
    total = len(results)
    acc = cnt / total if total > 0 else 0.0
    print(f"Accuracy: {cnt} / {total} = {acc :.4f}")

    results.append({"accuracy": acc})

    if args.log_samples and args.output_path:
        out_dir = os.path.dirname(args.output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        ext = os.path.splitext(args.output_path)[1].lower()
        if ext == ".jsonl":
            with open(args.output_path, "w") as f:
                for rec in results[:-1]:
                    f.write(json.dumps(rec) + "\n")
                f.write(json.dumps(results[-1]) + "\n")
        else:
            with open(args.output_path, "w") as f:
                json.dump(results, f, indent=4)
        print(f"Results saved to {args.output_path}")

    if len(acc_series) > 1:
        print("Accuracy trend:")
        print(ascii_sparkline(acc_series, max_len=80))


if __name__ == "__main__":
    main()

