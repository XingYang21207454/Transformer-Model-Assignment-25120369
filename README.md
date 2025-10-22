# Transformer Sentiment Classification 邢杨 25120369

This project implements a Transformer model from scratch (encoder + decoder) 
for sentiment classification on the `waimai_10k` dataset.

## 🧠 Features
- Handwritten Transformer architecture in PyTorch
- Includes positional encoding, self-attention, FFN, and residual connections
- Ablation experiments for major components
- Training stability with AdamW, scheduler, and gradient clipping

## 🚀 Quick Start

```bash
git clone https://github.com/<yourname>/transformer-sentiment.git
cd transformer-sentiment
pip install -r requirements.txt
bash scripts/run.sh

或在配置完环境后，直接运行src中的train.py
