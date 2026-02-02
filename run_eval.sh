#!/bin/bash
set -e


MODELS=(
  "allenai/OLMoE-1B-7B-0924-Instruct"
  "allenai/OLMoE-1B-7B-0125-Instruct"
)

for MODEL in "${MODELS[@]}"; do
  echo "======================================"
  echo "Evaluating model: $MODEL"
  echo "======================================"

  python diagnose-moe-gpt.py --model_name "$MODEL"

  echo "Finished: $MODEL"
  echo
done
