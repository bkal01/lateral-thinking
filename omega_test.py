from datasets import load_dataset

dataset = load_dataset("allenai/omega-transformative", "trans_circles")

train_ds = dataset["train"]
test_ds = dataset["test"]

counter = 0
total_samples = len(test_ds)

for i, sample in enumerate(test_ds):
    messages = sample.get("messages", [])
    
    # Convert messages to string for searching
    messages_text = str(messages)
    
    keywords = ["What", "Find", "Compute", "Determine", "Calculate", "Estimate", "Solve", "Estimate", "Calculate", "Determine", "Solve"]
    if not any(keyword in messages_text for keyword in keywords):
        print(f"Sample {i}:")
        print(sample)
        print("-" * 50)
        counter += 1

print(f"\nTotal samples without keywords: {counter}")
print(f"Total samples in dataset: {total_samples}")
print(f"Percentage: {(counter / total_samples) * 100:.2f}%")