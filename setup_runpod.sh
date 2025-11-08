#!/bin/bash
# Quick setup script for RunPod A6000 instance
# Run with: bash setup_runpod.sh

set -e  # Exit on error

echo "🚀 M4 Training Setup for A6000"
echo "================================"

# Check GPU
echo ""
echo "📊 Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if nvidia-smi | grep -q "A6000"; then
    echo "✅ A6000 detected!"
else
    echo "⚠️  Warning: Not an A6000. Adjust batch sizes if needed."
fi

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt
echo "✅ Dependencies installed"

# Verify installations
echo ""
echo "🔍 Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'BF16 support: {torch.cuda.is_bf16_supported()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
echo "✅ All core packages working"

# Check M4 data
echo ""
echo "📂 Checking M4 dataset..."
if [ -d "M4/data" ]; then
    num_files=$(ls M4/data/*.jsonl 2>/dev/null | wc -l)
    echo "✅ M4 data found: $num_files JSONL files"
    echo "Available generators:"
    ls M4/data/*.jsonl | head -5 | xargs -n1 basename
else
    echo "⚠️  M4 data not found. Make sure to clone/download M4 dataset."
fi

# W&B setup
echo ""
read -p "🔐 Setup Weights & Biases? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    wandb login
    echo "✅ W&B configured"
else
    echo "⏭️  Skipped W&B setup"
fi

# Quick test
echo ""
read -p "🧪 Run quick test? (10 min, verifies everything works) (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running quick test..."
    python train.py \
      --model roberta-base \
      --lr 2e-5 \
      --dropout 0.2 \
      --val_generator flant5 \
      --epochs 1 \
      --batch_size 64 \
      --max_train_samples 1000 \
      --max_val_samples 200 \
      --bf16 \
      --output_dir runs/setup-test

    echo ""
    echo "✅ Test completed! Check output above for any errors."
else
    echo "⏭️  Skipped test"
fi

# Final instructions
echo ""
echo "================================"
echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Review example_commands.sh for training commands"
echo "2. Run recommended training:"
echo ""
echo "   python train.py \\"
echo "     --model roberta-base \\"
echo "     --lr 2e-5 \\"
echo "     --dropout 0.2 \\"
echo "     --data_dir M4_cleaned/data \\"
echo "     --reddit_file reddit_training_60k.jsonl \\"
echo "     --epochs 3 \\"
echo "     --batch_size 64 \\"
echo "     --bf16 \\"
echo "     --stratified_split \\"
echo "     --validation_split 0.1 \\"
echo "     --use_class_weights \\"
echo "     --use_wandb \\"
echo "     --run_name m4-a6000 \\"
echo "     --output_dir runs/m4-a6000"
echo ""
echo "⏱️  Expected time: 4-6 hours"
echo "🎯 Expected F1: ~94-95%"
echo "📊 Monitor: https://wandb.ai (if using W&B)"
echo ""
