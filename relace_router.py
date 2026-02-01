from transformers import AutoModelForCausalLM
import torch

# Load both models
bad_model = AutoModelForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924-Instruct")
good_model = AutoModelForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0125-Instruct")

# Swap routers at each layer
for layer_idx in range(len(bad_model.model.layers)):
    bad_layer = bad_model.model.layers[layer_idx]
    good_layer = good_model.model.layers[layer_idx]
    breakpoint()
    
    # Replace the router (gate)
    bad_layer.mlp.gate.weight.data = good_layer.mlp.gate.weight.data.clone()
    if bad_layer.mlp.gate.bias is not None:
        bad_layer.mlp.gate.bias.data = good_layer.mlp.gate.bias.data.clone()

print("Router swap complete!")