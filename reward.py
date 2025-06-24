import re

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

def correctness_reward(completion: str, answer: str, **kwargs) -> float:
    """
    Reward for correctness in GSM8K. 1.0 if the completion answer is correct, 0.0 otherwise.
    """
    completion_answer = extract_gsm8k_answer(completion)
    answer_answer = extract_gsm8k_answer(answer)
    return float(completion_answer == answer_answer)

def main():
    print(correctness_reward("The answer is 10.", "The answer is 10."))

if __name__ == "__main__":
    main()