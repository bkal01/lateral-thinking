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


def chat(model_name: str, cfg: Dict[str, Any], prompt: str) -> None:
    """Generate a response to a prompt, showing thinking steps."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )

    messages = [{"role": "user", "content": prompt}]
    
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    
    inputs = tokenizer([formatted_prompt], return_tensors="pt").to(model.device)
    
    gen_kwargs = {k: cfg[k] for k in [
        "max_new_tokens",
        "temperature", 
        "top_p",
        "top_k",
        "min_p",
        "do_sample",
    ] if k in cfg}
    
    outputs = model.generate(**inputs, **gen_kwargs)
    gen_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
    
    full_response = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()
    
    print(f"Prompt: {prompt}\n")
    print(f"Response: {full_response}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a HF causal LM on a dataset or chat with it.")
    parser.add_argument("--model", required=True, help="HuggingFace model id")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--prompt", required=True, help="Prompt to send to model")
    
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    chat(args.model, cfg, args.prompt)


if __name__ == "__main__":
    main()
