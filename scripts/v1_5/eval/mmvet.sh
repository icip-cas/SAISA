#!/bin/bash

python -m llava.eval.model_vqa \
    --model-path yuanqianhao/saisa-vicuna \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/saisa-vicuna.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/saisa-vicuna.jsonl \
    --dst ./playground/data/eval/mm-vet/results/saisa-vicuna.json

