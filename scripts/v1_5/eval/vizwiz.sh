#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path yuanqianhao/saisa-vicuna \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ./playground/data/eval/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/saisa-vicuna.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/saisa-vicuna.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/saisa-vicuna.json
