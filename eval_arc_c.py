import torch
import random
import os
import re
import json
from tqdm import tqdm
from datasets import Dataset
from collections import Counter
from src.transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    OlmoeForCausalLM,
    # StoppingCriteriaList
)
from src.transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM
from datasets import load_dataset


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
    model = OlmoeForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
    )
    return tokenizer, model


def load_arc_c_dataset(split="test"):
    print("\nLoading BOOLQ dataset...")
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    print(f"BOOLQ {split} size: {len(dataset)}")
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
            max_new_tokens=3,
            # pad_token_id=tokenizer.eos_token_id,
            # stopping_criteria=stop_criteria
        )
    # breakpoint()
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract text after 'A:'
    # output_text = output_text.split("Answer:")[-1].strip()
    # print(f"The output text is {output_text}")
    return output_text

def extract_option(s: str) -> str | None:
    match = re.search(r'\b([A-D])\b', s)
    return match.group(1) if match else None


def evaluate_single_example(model, tokenizer, example, temp=0.0):
    """Evaluate a single GSM8K example."""
    # Prepare input
    labels = example["choices"]["label"]
    texts = example["choices"]["text"]

    choices_str = "\n".join(
        f"{l}. {t}" for l, t in zip(labels, texts)
    )

    input_text = f"""
    Question: {example['question']}

    Choices:
    {choices_str}

    Select the one correct option (A, B, C, or D) and output only the letter.

    Answer:
    """.strip()
    breakpoint()

    # Ground truth
    ground_truth = example['answerKey']

    output_text = generate_answer(model, tokenizer, input_text)
    # breakpoint()
    answer = output_text.split("Answer:")[-1].strip()
    extracted_answer = extract_option(answer)
    # breakpoint()
    correct = (extracted_answer == ground_truth)

    return {
        'question': example['question'] +  str(example['choices']),
        'gold_answer_text': example['answerKey'],
        'model_answers_text': output_text,
        'extracted_model_answers': extracted_answer,
        'extracted_gold_answer': ground_truth,
        'correct': correct
    }


# -----------------------------
# 3️⃣ Full evaluation loop
# -----------------------------

def evaluate_model_on_arc_c(model, tokenizer, dataset, temp=0.0):
    """Evaluate the model on the GSM8K test set."""
    results = []
    for index, example in tqdm(enumerate(dataset), desc="Evaluating GSM8K"):
        print(example)
        result = evaluate_single_example(
            model, tokenizer, example,
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

def save_results(results, output_file=None):
    """
    Save evaluation results to a JSON file.
    
    Args:
        results: Evaluation results dictionary
        swap_config: Dictionary containing swap configuration
        total_time: Total evaluation time
        output_file: Path to output file (optional)
    """

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    import argparse
    import time
    import gc
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Hugging Face model repo to evaluate"
    )

    args = parser.parse_args()

    # set seed for reprod
    set_seed(42)

    print(f"\n===== Evaluating {args.model_name} =====")

    dataset = load_arc_c_dataset(split="test")
    dataset = Dataset.from_dict(dataset[:500])

    start_time = time.time()

    tokenizer, model = load_model_and_tokenizer(args.model_name)
    model.eval()

    with torch.no_grad():
        results = evaluate_model_on_arc_c(
            model,
            tokenizer,
            dataset,

        )

    
    short_name = args.model_name.split("/")[-1]
    output_file = (
        f"/teamspace/studios/this_studio/moe-diagnose/output/arc_c/"
        f"{short_name}.json"
    )
    save_results(results, output_file)

    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f}s")

    # 🔥 HARD CLEANUP (important)
    del model, tokenizer, results
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    