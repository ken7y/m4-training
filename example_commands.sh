#!/bin/bash
# M4 Training Commands - Optimized for A6000 (48GB VRAM)

# ============================================
# 1. SETUP (Run once on new runpod instance)
# ============================================

# Install dependencies
pip install -r requirements.txt

# Verify GPU (should show A6000 with 48GB)
nvidia-smi

# Login to Weights & Biases (recommended for tracking)
wandb login

# ============================================
# 2. RECOMMENDED: Full Fine-Tuning on A6000
# ============================================

# This is the best approach for A6000 - uses its full power
python train.py \
  --model roberta-base \
  --lr 2e-5 \
  --dropout 0.2 \
  --epochs 3 \
  --batch_size 64 \
  --data_dir M4_cleaned/data \
  --reddit_file reddit_training_60k.jsonl \
  --bf16 \
  --stratified_split \
  --validation_split 0.1 \
  --use_class_weights \
  --use_wandb \
  --run_name m4-a6000-base \
  --output_dir runs/m4-a6000-base

# Expected: ~94-95% F1, ~4-6 hours training time

# ============================================
# 3. AGGRESSIVE: RoBERTa-Large (Better Performance)
# ============================================

# Use the A6000's memory for a larger model
python train.py \
  --model roberta-large \
  --lr 1e-5 \
  --dropout 0.2 \
  --epochs 3 \
  --batch_size 32 \
  --data_dir M4_cleaned/data \
  --reddit_file reddit_training_60k.jsonl \
  --bf16 \
  --stratified_split \
  --validation_split 0.1 \
  --use_class_weights \
  --use_wandb \
  --run_name m4-a6000-large \
  --output_dir runs/m4-a6000-large

# Expected: ~95-96% F1, ~8-10 hours training time

# ============================================
# 4. MULTI-DOMAIN TRAINING (Full Dataset)
# ============================================

python train.py \
  --model roberta-base \
  --lr 2e-5 \
  --dropout 0.2 \
  --train_domains arxiv wikipedia reddit peerread \
  --train_generators chatGPT davinci cohere dolly flant5 \
  --epochs 3 \
  --batch_size 64 \
  --data_dir M4_cleaned/data \
  --reddit_file reddit_training_60k.jsonl \
  --bf16 \
  --stratified_split \
  --validation_split 0.1 \
  --use_class_weights \
  --use_wandb \
  --run_name m4-multidomain \
  --output_dir runs/m4-multidomain

# ============================================
# 5. QUICK TEST (Verify Setup)
# ============================================

# Run this first to verify everything works (takes ~10 min)
python train.py \
  --model roberta-base \
  --lr 2e-5 \
  --dropout 0.2 \
  --epochs 1 \
  --batch_size 64 \
  --max_train_samples 5000 \
  --max_val_samples 1000 \
  --data_dir M4_cleaned/data \
  --reddit_file reddit_training_60k.jsonl \
  --bf16 \
  --stratified_split \
  --validation_split 0.1 \
  --use_class_weights \
  --output_dir runs/test

# ============================================
# 6. INFERENCE
# ============================================

# Run predictions on held-out data
python predict.py \
  --model_path runs/m4-a6000-base \
  --input_file M4/data/arxiv_chatGPT.jsonl \
  --output_file predictions.jsonl \
  --batch_size 64

# ============================================
# 7. EXPERIMENT: Different Validation Generators
# ============================================

# Test generalization to different unseen generators
for generator in flant5 dolly llama; do
  python train.py \
    --model roberta-base \
    --lr 2e-5 \
    --dropout 0.2 \
    --val_generator $generator \
    --epochs 3 \
    --batch_size 64 \
    --bf16 \
    --use_wandb \
    --run_name m4-val-$generator \
    --output_dir runs/m4-val-$generator
done

# ============================================
# A6000-SPECIFIC NOTES:
# ============================================
# ✅ ALWAYS use --bf16 (better than fp16 on Ampere)
# ✅ ALWAYS use batch_size 64 for roberta-base (fully utilizes GPU)
# ✅ NO NEED to freeze layers (you have 48GB, use it!)
# ✅ Learning rate 2e-5 is proven optimal for full fine-tuning
# ⚡ Expected training time: 4-6 hours for full M4 dataset
# 📊 Use W&B to monitor: train/val F1, per-generator metrics
#
# Available generators in M4/data (check with: ls M4/data/*.jsonl | head):
#   chatGPT, davinci, cohere, bloomz, dolly, flant5, llama
#
# Memory usage estimates (batch 64, bf16):
#   roberta-base:  ~18GB / 48GB (37% utilization)
#   roberta-large: ~35GB / 48GB (73% utilization)
#   deberta-large: ~42GB / 48GB (87% utilization)
