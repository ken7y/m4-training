#!/bin/bash
# RunPod Setup Script - Run this first on your RunPod instance
set -e

echo "=================================="
echo "M4 TRAINING - RUNPOD SETUP"
echo "=================================="

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt

# Verify data exists
echo ""
echo "📊 Verifying data..."

if [ ! -d "M4_cleaned/data" ]; then
    echo "❌ ERROR: M4_cleaned/data directory not found!"
    echo "   Did you upload the cleaned data?"
    exit 1
fi

if [ ! -f "reddit_training_60k.jsonl" ]; then
    echo "❌ ERROR: reddit_training_60k.jsonl not found!"
    echo "   Did you upload the Reddit data?"
    exit 1
fi

# Count files
M4_FILES=$(ls M4_cleaned/data/*.jsonl 2>/dev/null | wc -l)
M4_LINES=$(wc -l M4_cleaned/data/*.jsonl 2>/dev/null | tail -1 | awk '{print $1}')
REDDIT_LINES=$(wc -l reddit_training_60k.jsonl 2>/dev/null | awk '{print $1}')

echo "✅ M4 cleaned data: $M4_FILES files, $M4_LINES lines"
echo "✅ Reddit data: $REDDIT_LINES lines"

# Quick validation test
echo ""
echo "🧪 Running validation test..."
python3 -c "
import sys
sys.path.insert(0, '.')
from train import load_m4_data, load_reddit_data

# Test M4 data
m4_data = load_m4_data('M4_cleaned/data', ['arxiv'], ['chatGPT'], max_samples=10)
print(f'✅ M4 data loads correctly: {len(m4_data)} samples')

# Test Reddit data
reddit_data = load_reddit_data('reddit_training_60k.jsonl', max_samples=10)
print(f'✅ Reddit data loads correctly: {len(reddit_data)} samples')
"

# Optional: W&B login
echo ""
echo "🔑 Weights & Biases setup (optional):"
echo "   Run: wandb login"
echo "   Or skip if you don't want tracking"

echo ""
echo "=================================="
echo "✅ SETUP COMPLETE!"
echo "=================================="
echo ""
echo "Next step: Run the training script"
echo "  ./start_training.sh"
echo ""
