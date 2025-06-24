import argparse
import re
from pathlib import Path
from typing import Any, Dict

import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from tqdm import tqdm


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file into a dictionary."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_gsm8k_answer(text: str) -> str:
    """Return the numeric answer from a model response for GSM8K.

    The heuristic is:
    1. If there is a pattern like "#### <answer>", return the number after `####`.
    2. Otherwise, return the last standalone number in the text.
    """
    match = re.search(r"####\s*(-?\d+\.?\d*)", text)
    if match:
        return match.group(1)

    numbers = re.findall(r"-?\d+\.?\d*", text)
    return numbers[-1] if numbers else ""


def evaluate(model_name: str, cfg: Dict[str, Any], dataset_spec: str) -> None:
    """Run evaluation and print accuracy."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )

    parts = dataset_spec.split(":")
    ds_name = parts[0]
    ds_cfg = parts[1] if len(parts) > 1 else None
    ds_split = parts[2] if len(parts) > 2 else "test"

    dataset = (
        load_dataset(ds_name, ds_cfg, split=ds_split)
        if ds_cfg
        else load_dataset(ds_name, split=ds_split)
    )

    correct = 0
    total = len(dataset)

    print(f"Evaluating {model_name} on {total} examples from {dataset_spec}")
    print("-" * 60)

    # try to find the </think> token id if it exists
    try:
        think_token_id = tokenizer.convert_tokens_to_ids("</think>")
        if think_token_id == tokenizer.unk_token_id:
            think_token_id = None
    except Exception:
        think_token_id = None

    gen_kwargs = {k: cfg[k] for k in [
        "max_new_tokens",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "do_sample",
    ] if k in cfg}

    for i, ex in tqdm(enumerate(dataset), total=total):
        if ds_name == "gsm8k":
            question = ex["question"].strip()
            messages = [{"role": "user", "content": question}]

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

            outputs = model.generate(**inputs, **gen_kwargs)
            gen_ids = outputs[0][len(inputs.input_ids[0]):].tolist()

            if think_token_id is not None:
                try:
                    idx = len(gen_ids) - gen_ids[::-1].index(think_token_id)
                except ValueError:
                    idx = 0
                answer_ids = gen_ids[idx:]
            else:
                answer_ids = gen_ids

            answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
            pred = extract_gsm8k_answer(answer_text)
            gold = extract_gsm8k_answer(ex["answer"].strip())
            print(f"Question: {question}\n")
            print(f"Answer: {answer_text}\n")
            print(f"Pred: {pred}, Gold: {gold}\n")
            if pred == gold:
                correct += 1
        else:
            raise NotImplementedError(f"Dataset '{ds_name}' not supported yet.")

    print("-" * 60)
    acc = correct / total if total else 0.0
    print(f"Final Accuracy: {acc * 100:.2f}% ({correct}/{total})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a HF causal LM on a dataset.")
    parser.add_argument("--model", required=True, help="HuggingFace model id")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--dataset", required=True, help="Dataset spec, e.g. 'gsm8k:main:test'")
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate(args.model, cfg, args.dataset)


if __name__ == "__main__":
    main()
