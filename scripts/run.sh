#!/bin/bash
# 设置工作目录为项目根目录
cd "$(dirname "$0")/.." || exit

# 创建结果目录
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

echo "==============================================="
echo "Starting Transformer Sentiment Classification"
echo "==============================================="
echo ""

# 检查 Python 环境
if ! command -v python &> /dev/null
then
    echo "Python not found! Please activate your environment first."
    exit
fi

# 运行训练脚本
echo "Running training script..."
python src/transformer_sentiment.py > "$RESULTS_DIR/train_log.txt" 2>&1

# 检查是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "Training completed successfully!"
    echo "Log file saved at: $RESULTS_DIR/train_log.txt"
    echo "If your script saves figures, they should appear under: $RESULTS_DIR/"
else
    echo ""
    echo "Training failed. Check $RESULTS_DIR/train_log.txt for details."
fi

echo ""
echo "==============================================="
echo "Done."
echo "==============================================="
