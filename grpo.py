import argparse
from pathlib import Path
from typing import Any, Dict, List

import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

import verifiers as vf

from reward import correctness_reward

parser = vf.parsers.XMLParser(["think", "answer"])

rubric = vf.rubrics.Rubric(
    funcs=[correctness_reward],
    weights=[1.0],
    parser=parser,
)

def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file into a dictionary."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def evaluate(model_name: str, cfg: Dict[str, Any], dataset_spec: str) -> None:
    """Run evaluation and print average reward using the correctness rubric."""
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

    rewards: List[float] = []
    records: List[Dict[str, str | float]] = []

    print(f"Evaluating {model_name} on {len(dataset)} examples from {dataset_spec}")
    print("-" * 60)

    for ex in tqdm(dataset, total=len(dataset)):
        if ds_name != "gsm8k":
            raise NotImplementedError(f"Dataset '{ds_name}' not supported yet.")

        question = ex["question"].strip()
        gold_answer = ex["answer"].strip()

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
                reasoning_ids = gen_ids[:idx-1]
                reasoning_trace = tokenizer.decode(reasoning_ids, skip_special_tokens=True).strip()
            except ValueError:
                idx = 0
                reasoning_trace = ""
            answer_ids = gen_ids[idx:]
        else:
            reasoning_trace = ""
            answer_ids = gen_ids

        completion = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

        reward = correctness_reward(completion, gold_answer)
        rewards.append(reward)

        records.append({
            "prompt": question,
            "completion": completion,
            "answer": gold_answer,
            "reward": reward,
        })
        print(records[-1])

    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

    print("-" * 60)
    print(f"Average reward: {avg_reward:.4f}")

def main() -> None:
    parser_ = argparse.ArgumentParser(
        description="Evaluate a HF causal LM using the correctness reward rubric.")
    parser_.add_argument("--model", required=True, help="HuggingFace model id")
    parser_.add_argument("--config", required=True, help="Path to YAML config file")
    parser_.add_argument("--dataset", required=True, help="Dataset spec, e.g. 'gsm8k:main:test'")
    args = parser_.parse_args()

    cfg = load_config(args.config)
    evaluate(args.model, cfg, args.dataset)


if __name__ == "__main__":
    main()
