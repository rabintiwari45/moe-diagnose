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
    model = OlmoeForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
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

def generate_answer(model, tokenizer, input_text, save_name):
    """Generate model output for a single question."""
    inputs = tokenizer(input_text, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            save_name=save_name
            # pad_token_id=tokenizer.eos_token_id,
            # stopping_criteria=stop_criteria
        )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract text after 'A:'
    output_text = output_text.split("A:")[-1].strip()
    print(f"The output text is {output_text}")
    return output_text


def evaluate_single_example(model, tokenizer, save_name, example, use_cot_prompt=False,
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
            output_text = generate_answer(model, tokenizer, input_text, save_name)
            numeric = extract_predicted_answer(output_text)
            model_answers.append({'text': output_text, 'numeric': numeric})
    else:
        output_text = generate_answer(model, tokenizer, input_text, save_name)
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

def evaluate_model_on_gsm8k(model, tokenizer, model_name, dataset, use_cot_prompt=False,
                            use_majority_vote=False, n_votes=1, temp=0.0):
    """Evaluate the model on the GSM8K test set."""
    results = []
    for index, example in tqdm(enumerate(dataset), desc="Evaluating GSM8K"):
        print(example)
        name = model_name.split("/")[1]
        if use_cot_prompt:
            name += "_with_cot"
        else:
            name += "_without_cot"
        name += f"_{index}_selected_experts.pkl"

        result = evaluate_single_example(
            model, tokenizer, name, example,
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



def swap_router_attention_mlp_experts(bad_model, good_model):
    """
    Swap MLP expert weights from good_model to bad_model.
    
    Args:
        bad_model: The model whose MLP experts will be replaced
        good_model: The model whose MLP experts will be copied from
    
    Returns:
        bad_model: Modified model with swapped MLP experts
    """
    # Swap MLP experts at each layer
    for layer_idx in range(len(bad_model.model.layers)):
        bad_layer = bad_model.model.layers[layer_idx]
        good_layer = good_model.model.layers[layer_idx]

        bad_layer.mlp.gate.weight.data = good_layer.mlp.gate.weight.data.clone()
        if bad_layer.mlp.gate.bias is not None:
            bad_layer.mlp.gate.bias.data = good_layer.mlp.gate.bias.data.clone()

        # Replace q_proj
        bad_layer.self_attn.q_proj.weight.data = good_layer.self_attn.q_proj.weight.data.clone()
        if bad_layer.self_attn.q_proj.bias is not None:
            bad_layer.self_attn.q_proj.bias.data = good_layer.self_attn.q_proj.bias.data.clone()
        
        # Replace k_proj
        bad_layer.self_attn.k_proj.weight.data = good_layer.self_attn.k_proj.weight.data.clone()
        if bad_layer.self_attn.k_proj.bias is not None:
            bad_layer.self_attn.k_proj.bias.data = good_layer.self_attn.k_proj.bias.data.clone()
        
        # Replace v_proj
        bad_layer.self_attn.v_proj.weight.data = good_layer.self_attn.v_proj.weight.data.clone()
        if bad_layer.self_attn.v_proj.bias is not None:
            bad_layer.self_attn.v_proj.bias.data = good_layer.self_attn.v_proj.bias.data.clone()
        
        # Replace o_proj
        bad_layer.self_attn.o_proj.weight.data = good_layer.self_attn.o_proj.weight.data.clone()
        if bad_layer.self_attn.o_proj.bias is not None:
            bad_layer.self_attn.o_proj.bias.data = good_layer.self_attn.o_proj.bias.data.clone()
        
        # Replace q_norm
        bad_layer.self_attn.q_norm.weight.data = good_layer.self_attn.q_norm.weight.data.clone()
        if hasattr(bad_layer.self_attn.q_norm, 'bias') and bad_layer.self_attn.q_norm.bias is not None:
            bad_layer.self_attn.q_norm.bias.data = good_layer.self_attn.q_norm.bias.data.clone()
        
        # Replace k_norm
        bad_layer.self_attn.k_norm.weight.data = good_layer.self_attn.k_norm.weight.data.clone()
        if hasattr(bad_layer.self_attn.k_norm, 'bias') and bad_layer.self_attn.k_norm.bias is not None:
            bad_layer.self_attn.k_norm.bias.data = good_layer.self_attn.k_norm.bias.data.clone()
        
        # Swap all 64 experts
        for expert_idx in range(len(bad_layer.mlp.experts)):
            bad_expert = bad_layer.mlp.experts[expert_idx]
            good_expert = good_layer.mlp.experts[expert_idx]
            
            # Replace gate_proj
            bad_expert.gate_proj.weight.data = good_expert.gate_proj.weight.data.clone()
            if bad_expert.gate_proj.bias is not None:
                bad_expert.gate_proj.bias.data = good_expert.gate_proj.bias.data.clone()
            
            # Replace up_proj
            bad_expert.up_proj.weight.data = good_expert.up_proj.weight.data.clone()
            if bad_expert.up_proj.bias is not None:
                bad_expert.up_proj.bias.data = good_expert.up_proj.bias.data.clone()
            
            # Replace down_proj
            bad_expert.down_proj.weight.data = good_expert.down_proj.weight.data.clone()
            if bad_expert.down_proj.bias is not None:
                bad_expert.down_proj.bias.data = good_expert.down_proj.bias.data.clone()
    
    return bad_model

def swap_model_components(bad_model, good_model, swap_router=True, swap_attention=True, swap_experts=True):
    """
    Selectively swap components from good_model to bad_model.
    
    Args:
        bad_model: The model whose components will be replaced
        good_model: The model whose components will be copied from
        swap_router: Whether to swap router (gate) weights
        swap_attention: Whether to swap attention weights
        swap_experts: Whether to swap MLP expert weights
    
    Returns:
        bad_model: Modified model with swapped components
    """
    print(f"Swapping components - Router: {swap_router}, Attention: {swap_attention}, Experts: {swap_experts}")
    
    # Swap components at each layer
    for layer_idx in range(len(bad_model.model.layers)):
        bad_layer = bad_model.model.layers[layer_idx]
        good_layer = good_model.model.layers[layer_idx]

        # Swap router (gate)
        if swap_router:
            bad_layer.mlp.gate.weight.data = good_layer.mlp.gate.weight.data.clone()
            if bad_layer.mlp.gate.bias is not None:
                bad_layer.mlp.gate.bias.data = good_layer.mlp.gate.bias.data.clone()

        # Swap attention components
        if swap_attention:
            # Replace q_proj
            bad_layer.self_attn.q_proj.weight.data = good_layer.self_attn.q_proj.weight.data.clone()
            if bad_layer.self_attn.q_proj.bias is not None:
                bad_layer.self_attn.q_proj.bias.data = good_layer.self_attn.q_proj.bias.data.clone()
            
            # Replace k_proj
            bad_layer.self_attn.k_proj.weight.data = good_layer.self_attn.k_proj.weight.data.clone()
            if bad_layer.self_attn.k_proj.bias is not None:
                bad_layer.self_attn.k_proj.bias.data = good_layer.self_attn.k_proj.bias.data.clone()
            
            # Replace v_proj
            bad_layer.self_attn.v_proj.weight.data = good_layer.self_attn.v_proj.weight.data.clone()
            if bad_layer.self_attn.v_proj.bias is not None:
                bad_layer.self_attn.v_proj.bias.data = good_layer.self_attn.v_proj.bias.data.clone()
            
            # Replace o_proj
            bad_layer.self_attn.o_proj.weight.data = good_layer.self_attn.o_proj.weight.data.clone()
            if bad_layer.self_attn.o_proj.bias is not None:
                bad_layer.self_attn.o_proj.bias.data = good_layer.self_attn.o_proj.bias.data.clone()
            
            # Replace q_norm
            bad_layer.self_attn.q_norm.weight.data = good_layer.self_attn.q_norm.weight.data.clone()
            if hasattr(bad_layer.self_attn.q_norm, 'bias') and bad_layer.self_attn.q_norm.bias is not None:
                bad_layer.self_attn.q_norm.bias.data = good_layer.self_attn.q_norm.bias.data.clone()
            
            # Replace k_norm
            bad_layer.self_attn.k_norm.weight.data = good_layer.self_attn.k_norm.weight.data.clone()
            if hasattr(bad_layer.self_attn.k_norm, 'bias') and bad_layer.self_attn.k_norm.bias is not None:
                bad_layer.self_attn.k_norm.bias.data = good_layer.self_attn.k_norm.bias.data.clone()
        
        # Swap MLP experts
        if swap_experts:
            # Swap all 64 experts
            for expert_idx in range(len(bad_layer.mlp.experts)):
                bad_expert = bad_layer.mlp.experts[expert_idx]
                good_expert = good_layer.mlp.experts[expert_idx]
                
                # Replace gate_proj
                bad_expert.gate_proj.weight.data = good_expert.gate_proj.weight.data.clone()
                if bad_expert.gate_proj.bias is not None:
                    bad_expert.gate_proj.bias.data = good_expert.gate_proj.bias.data.clone()
                
                # Replace up_proj
                bad_expert.up_proj.weight.data = good_expert.up_proj.weight.data.clone()
                if bad_expert.up_proj.bias is not None:
                    bad_expert.up_proj.bias.data = good_expert.up_proj.bias.data.clone()
                
                # Replace down_proj
                bad_expert.down_proj.weight.data = good_expert.down_proj.weight.data.clone()
                if bad_expert.down_proj.bias is not None:
                    bad_expert.down_proj.bias.data = good_expert.down_proj.bias.data.clone()
    
    return bad_model

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
    # 1. Setup
    good_model_name = "allenai/OLMoE-1B-7B-0125-Instruct"
    bad_model_name = "allenai/OLMoE-1B-7B-0924-Instruct"
    set_seed(42)
    model_names = [
        "rdabin/OLMoE-1B-7B-0924-Instruct-attention_only",
        "rdabin/OLMoE-1B-7B-0924-Instruct-router_only",
        "rdabin/OLMoE-1B-7B-0924-Instruct-experts_only"
    ]
    
    dataset = load_gsm8k_dataset(split="test")


    # Define all experiments to run
    experiments = [
        # Single component swaps
        {
            "name": "router_only",
            "swap_router": True,
            "swap_attention": False,
            "swap_experts": False
        },
        {
            "name": "attention_only",
            "swap_router": False,
            "swap_attention": True,
            "swap_experts": False
        },
        {
            "name": "experts_only",
            "swap_router": False,
            "swap_attention": False,
            "swap_experts": True
        },
        # Two component combinations
        {
            "name": "router_and_attention",
            "swap_router": True,
            "swap_attention": True,
            "swap_experts": False
        },
        {
            "name": "router_and_experts",
            "swap_router": True,
            "swap_attention": False,
            "swap_experts": True
        },
        {
            "name": "attention_and_experts",
            "swap_router": False,
            "swap_attention": True,
            "swap_experts": True
        },
        # All components
        {
            "name": "all_components",
            "swap_router": True,
            "swap_attention": True,
            "swap_experts": True
        },

        # bad model
        {
            "name": "bad_model",
            "swap_router": False,
            "swap_attention": False,
            "swap_experts": False
        },
        # good model
        {
            "name": "good_model",
            "swap_router": True,
            "swap_attention": True,
            "swap_experts": True
        },
 
    ]

    for model_name in model_names:
        import time
        start_time = time.time()
        tokenizer, model = load_model_and_tokenizer(model_name)
        results = evaluate_model_on_gsm8k(
            model,
            tokenizer,
            bad_model_name,
            Dataset.from_dict(dataset[:10]),  # for testing small subset first
            use_cot_prompt=True,
            use_majority_vote=False
        )
        model_name = model_name.split("/")[-1]
        output_file = f"/content/drive/MyDrive/output/{model_name}.json"
        save_results(results, output_file)
        total_time =time.time() - start_time
        print(f"The total time is {total_time}")
    