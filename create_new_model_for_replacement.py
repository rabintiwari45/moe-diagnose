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

def main(swap_router, swap_attention, swap_experts, name):
    good_model_name = "allenai/OLMoE-1B-7B-0125-Instruct"
    bad_model_name = "allenai/OLMoE-1B-7B-0924-Instruct"
    good_tokenizer, good_model = load_model_and_tokenizer(good_model_name)
    bad_tokenizer, bad_model = load_model_and_tokenizer(bad_model_name)


    model = swap_model_components(bad_model, 
                    good_model,
                    swap_router=swap_router,
                    swap_attention=swap_attention,
                    swap_experts=swap_experts)
    path = f"rdabin/OLMoE-1B-7B-0924-Instruct-{name}"

    from huggingface_hub import login
    login(token="")

    model.push_to_hub(path)
    bad_tokenizer.push_to_hub(path)

    return None


if __name__ == "__main__":
    experiments = [
        # Single component swaps
        # {
        #     "name": "router_only",
        #     "swap_router": True,
        #     "swap_attention": False,
        #     "swap_experts": False
        # },
        # {
        #     "name": "attention_only",
        #     "swap_router": False,
        #     "swap_attention": True,
        #     "swap_experts": False
        # },
        # {
        #     "name": "experts_only",
        #     "swap_router": False,
        #     "swap_attention": False,
        #     "swap_experts": True
        # },
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
        
        ]

    for exp in experiments:
        main(exp["swap_router"], exp["swap_attention"], exp["swap_experts"], exp["name"])
    