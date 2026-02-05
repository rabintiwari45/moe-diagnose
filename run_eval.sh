#!/bin/bash
set -e


MODELS=(
  "allenai/OLMoE-1B-7B-0125-Instruct"
  "allenai/OLMoE-1B-7B-0924-Instruct"

)

for MODEL in "${MODELS[@]}"; do
  echo "======================================"
  echo "Evaluating model: $MODEL"
  echo "======================================"

  python eval_arc_c.py --model_name "$MODEL"

  echo "Finished: $MODEL"
  echo
done
