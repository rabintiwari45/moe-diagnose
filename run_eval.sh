#!/bin/bash
set -e


MODELS=(
  "allenai/OLMoE-1B-7B-0924-Instruct"
  "allenai/OLMoE-1B-7B-0125-Instruct"
  "rdabin/OLMoE-1B-7B-0924-Instruct-router_only"
  "rdabin/OLMoE-1B-7B-0924-Instruct-attention_only"
  "rdabin/OLMoE-1B-7B-0924-Instruct-experts_only"

)

for MODEL in "${MODELS[@]}"; do
  echo "======================================"
  echo "Evaluating model: $MODEL"
  echo "======================================"

  python eval_hellaswag.py --model_name "$MODEL"

  echo "Finished: $MODEL"
  echo
done
