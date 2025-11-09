#!/bin/bash
# Start Training - Balanced M4 + Reddit with optimal settings
set -e

echo "=================================="
echo "M4 TRAINING - STARTING"
echo "=================================="
echo ""
echo "Configuration:"
echo "  • Dataset: M4 cleaned (63k lines) + Reddit (60k)"
echo "  • Total samples: ~186k (66% human, 34% AI)"
echo "  • Strategy: Weighted loss + stratified split"
echo "  • Expected time: 6-8 hours on A6000"
echo "  • Expected F1: 92-95%"
echo ""
echo "Press Ctrl+C now to cancel, or wait 5 seconds to start..."
sleep 5

echo ""
echo "🚀 Starting training..."
echo ""

python3 train.py \
  --data_dir M4_cleaned/data \
  --reddit_file reddit_training_60k.jsonl \
  --use_class_weights \
  --stratified_split \
  --validation_split 0.1 \
  --train_domains arxiv wikipedia reddit wikihow peerread \
  --train_generators chatGPT davinci cohere dolly flant5 \
  --epochs 3 \
  --batch_size 64 \
  --lr 2e-5 \
  --dropout 0.2 \
  --bf16 \
  --use_wandb \
  --run_name balanced-m4-reddit \
  --output_dir runs/balanced-m4-reddit

echo ""
echo "=================================="
echo "✅ TRAINING COMPLETE!"
echo "=================================="
echo ""
echo "Model saved to: runs/balanced-m4-reddit/"
echo ""
echo "Next steps:"
echo "  1. Download model: scp -r runs/balanced-m4-reddit/ local-machine:~/"
echo "  2. Test model: python3 predict.py --model_path runs/balanced-m4-reddit"
echo ""
