#!/bin/bash

python -m llava.eval.model_vqa_science \
    --model-path yuanqianhao/saisa-vicuna \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/saisa-vicuna.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/saisa-vicuna.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/saisa-vicuna_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/saisa-vicuna_result.json
