from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

model_name = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

subset = load_dataset("gsm8k", "main", split="train[:10]")

for idx, example in enumerate(subset):
    question = example["question"].strip()

    messages = [
        {"role": "user", "content": question}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # generate output according to Qwen Best Practices
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        do_sample=True
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # find </think> token index in the output
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print(f"Problem {idx + 1}: {question}")
    print("thinking content:", thinking_content)
    print("content:", content)
    print("ground truth:", example["answer"].strip())
    print("-" * 80)
