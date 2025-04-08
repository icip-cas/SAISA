# SAISA
Code release for "[SAISA: Towards Multimodal Large Language Models with Both Training and Inference Efficiency](https://arxiv.org/abs/2502.02458)"

## Install
1. Clone this repository and navigate to SAISA folder
```bash
git clone https://github.com/icip-cas/SAISA.git
cd SAISA
```

2. Install Package
```bash
conda create -n saisa python=3.10 -y
conda activate saisa
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for evaluation with lmms-eval
```bash
cd lmms_eval
pip install -e .
```

## Chatbot Inference
Chat about images using SAISA.

```bash
python -m llava.serve.cli \
    --model-path yuanqianhao/saisa-vicuna   \
    --image-file "https://llava-vl.github.io/static/images/view.jpg"
```

## SAISA Evaluation
### Evaluation with LMMs-Eval
[LMMs-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) is an evaluation framework meticulously crafted for consistent and efficient evaluation of LMM.

```bash
export MODEL_PATH="yuanqianhao/saisa-vicuna"
export MODEL_NAME="saisa_vicuna"
export CONV_MODE="v1"
accelerate launch  --num_processes=1 --main_process_port=12346 -m lmms_eval \
    --model llava \
    --model_args pretrained=${MODEL_PATH},conv_template=${CONV_MODE}  \
    --tasks mmmu_val \
    --batch_size 1 \
    --log_samples_suffix ${MODEL_NAME} \
    --output_path ./logs/ 
```

### Evaluation with Scripts From LLaVA

See [Evaluation.md](https://github.com/icip-cas/SAISA/blob/main/docs/Evaluation.md).

## Acknowledge
This work is built upon the [LLaVA](https://github.com/haotian-liu/LLaVA) and [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).

## Citation
If you find ShortV useful for your research and applications, please cite using this BibTeX:
```bib
@article{yuan2025saisa,
  title={SAISA: Towards Multimodal Large Language Models with Both Training and Inference Efficiency},
  author={Yuan, Qianhao and Liu, Yanjiang and Lu, Yaojie and Lin, Hongyu and He, Ben and Han, Xianpei and Sun, Le},
  journal={arXiv preprint arXiv:2502.02458},
  year={2025}
}
```