import torch
import random
import os
import re
import json
from tqdm import tqdm
from collections import Counter
from src.transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    # StoppingCriteriaList
)
from src.transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM
from datasets import load_dataset


def extract_predicted_answer(text):
    regex_pattern = "(-?[$0-9.,]{2,})|(-?[0-9]+)"
    regexes_to_ignore =[
        ",",
        "\\$",
        "(?s).*#### ",
        "\\.$"
    ]
    match = re.findall(regex_pattern, text)
    if match:
        match = match[-1]
        if isinstance(match, tuple):
            match = [m for m in match if m][0]
        text = match.strip()

        for regex in regexes_to_ignore:
            text = re.sub(regex, "", text)
        return text
    else:
        return None

def extract_ground_truth(text):
    return text.split('####')[-1].strip()

# -----------------------------
# 1️⃣ Setup functions
# -----------------------------

def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)
    return seed


def load_model_and_tokenizer(model_name):
    """Load the tokenizer and model."""
    print(f"Loading model and tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    )
    return tokenizer, model


def load_gsm8k_dataset(split='test'):
    """Load GSM8K dataset."""
    print("\nLoading GSM8K dataset...")
    dataset = load_dataset('gsm8k', "main", split=split)
    print(f"GSM8K {split} size: {len(dataset)}")
    return dataset


# -----------------------------
# 2️⃣ Generation helper
# -----------------------------

def generate_answer(model, tokenizer, input_text):
    """Generate model output for a single question."""
    inputs = tokenizer(input_text, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            # pad_token_id=tokenizer.eos_token_id,
            # stopping_criteria=stop_criteria
        )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract text after 'A:'
    output_text = output_text.split("A:")[-1].strip()
    return output_text


def evaluate_single_example(model, tokenizer, example, use_cot_prompt=False,
                            use_majority_vote=False, n_votes=1, temp=0.0):
    """Evaluate a single GSM8K example."""
    # Prepare input
    if use_cot_prompt:
        input_text = f"Q: {example['question']}\nA: Let's think step by step."
    else:
        input_text = f"Q: {example['question']}\nA:"

    # Ground truth
    ground_truth = extract_ground_truth(example['answer'])

    # Define stopping criteria
    # stop_words = ["Q:", "</s>", "<|im_end|>"]
    # stop_criteria = StoppingCriteriaList([
    #     SpecificStringStoppingCriteria(tokenizer, stop_words, len(input_text))
    # ])

    model_answers = []
    # Majority voting
    if use_majority_vote:
        for _ in range(n_votes):
            output_text = generate_answer(model, tokenizer, input_text)
            numeric = extract_predicted_answer(output_text)
            model_answers.append({'text': output_text, 'numeric': numeric})
    else:
        output_text = generate_answer(model, tokenizer, input_text)
        numeric = extract_predicted_answer(output_text)
        model_answers.append({'text': output_text, 'numeric': numeric})

    # Aggregate results
    numeric_answers = [ma['numeric'] for ma in model_answers]
    filtered = [num for num in numeric_answers if num is not None]
    majority_answer = Counter(filtered).most_common(1)[0][0] if filtered else None
    correct = (majority_answer == ground_truth) if majority_answer is not None else False

    return {
        'question': example['question'],
        'gold_answer_text': example['answer'],
        'model_answers_text': [ma['text'] for ma in model_answers],
        'extracted_model_answers': numeric_answers,
        'extracted_gold_answer': ground_truth,
        'majority_answer': majority_answer,
        'correct': correct
    }


# -----------------------------
# 3️⃣ Full evaluation loop
# -----------------------------

def evaluate_model_on_gsm8k(model, tokenizer, dataset, use_cot_prompt=False,
                            use_majority_vote=False, n_votes=1, temp=0.0):
    """Evaluate the model on the GSM8K test set."""
    results = []
    for example in tqdm(dataset, desc="Evaluating GSM8K"):
        print(example)
        result = evaluate_single_example(
            model, tokenizer, example,
            use_cot_prompt=use_cot_prompt,
            use_majority_vote=use_majority_vote,
            n_votes=n_votes,
            temp=temp
        )
        results.append(result)

    # Compute accuracy
    correct_count = sum(r['correct'] for r in results)
    total = len(results)
    accuracy = correct_count / total
    print(f"\nAccuracy: {correct_count} / {total} = {accuracy:.4f}")
    results.append({'accuracy': accuracy})
    return results


# -----------------------------
# 4️⃣ Save results
# -----------------------------

def save_results(results, model_name, use_cot_prompt=False,
                 use_majority_vote=False, n_votes=1, temp=0.0):
    """Save evaluation results to JSON."""
    os.makedirs('eval_results/zero_shot', exist_ok=True)
    short_name = model_name.split('/')[-1]
    result_file = f"eval_results/zero_shot/{short_name}"
    if use_cot_prompt:
        result_file += "_cot"
    if use_majority_vote:
        result_file += f"_maj1@{n_votes}_temp{temp}"
    result_file += "_results.json"

    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {result_file}")

if __name__ == "__main__":
    # 1. Setup
    model_name = "allenai/OLMoE-1B-7B-0125-Instruct"
    set_seed(42)

    tokenizer, model = load_model_and_tokenizer(model_name)
    dataset = load_gsm8k_dataset(split="test")
    from datasets import Dataset


    results = evaluate_model_on_gsm8k(
        model,
        tokenizer,
        Dataset.from_dict(dataset[:1]),  # for testing small subset first
        use_cot_prompt=True,
        use_majority_vote=False
    )