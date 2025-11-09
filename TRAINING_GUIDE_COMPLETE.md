# Complete Training Guide - M4 + Reddit Balanced Dataset

**Quick Start for RunPod Training**

---

## Dataset Overview

### What We Have
- **60k Reddit posts** (casual human text, pre-ChatGPT, already cleaned)
- **63k M4 samples** (formal text: academic papers, Wikipedia - now cleaned)

### Class Distribution (English Only)
```
M4 ENGLISH cleaned: 63,117 lines (only arxiv, wikipedia, wikihow, peerread, reddit)
  Each line = human_text (label=0) + machine_text (label=1)
  → 63,117 human + 63,117 AI from M4

Reddit 60k: 60,000 samples (all label=0, human, casual text)

TOTAL:
  Human: 123,117 (66.1%) - 60k casual + 63k formal
  AI:     63,117 (33.9%) - all formal
  ────────────────────────────────
  TOTAL: 186,234 samples
```

**Note**: Only ~2:1 imbalance (66% human, 34% AI) - this is MILD and manageable with weighted loss

---

## Solutions for Class Imbalance

### Option 1: Weighted Loss (RECOMMENDED)
Use weighted cross-entropy where AI class gets higher weight.

**Advantages:**
✅ Uses all data (no waste)  
✅ Easy to implement  
✅ Works well with RoBERTa  
✅ Standard practice  
✅ Already built into `train.py` when you pass `--use_class_weights`

**Implementation:**
```python
# In training script
from torch.nn import CrossEntropyLoss

# Calculate class weights (English only)
# M4 English: 63,117 human + 63,117 AI
# Reddit: 60,000 human
num_ai = 63117
total = num_human + num_ai

weight_human = total / (2 * num_human)  # 0.756
weight_ai = total / (2 * num_ai)        # 1.476
    --stratified_split \
    --validation_split 0.1 \

    --use_class_weights \
class_weights = torch.tensor([weight_human, weight_ai])
# AI loss weighted 1.95x higher than human

# Use in Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    # Add weighted loss
    compute_loss=lambda model, inputs, return_outputs=False: 
        weighted_loss(model, inputs, class_weights, return_outputs)
)
```

### Option 2: Undersample Human Text
Balance by randomly removing human samples.

**Result:**
```
Human: 63,117 (50%)
AI:    63,117 (50%)
Total: 126,234 samples
```

**Advantages:**
✅ Perfect balance  
✅ Simpler training  

**Disadvantages:**
❌ Wastes 60k human samples  
❌ Less diversity  

### Option 3: Oversample AI Text
Duplicate AI samples to match human count.

**Disadvantages:**
❌ Overfitting risk  
❌ Not recommended for text  

---

## Quick Start Commands

### 1. Clean M4 Data (if not done)
```bash
python3 clean_m4_data.py
```

### 2. Train with Weighted Loss & Stratified Split
```bash
python3 train.py \
  --data_dir M4_cleaned/data \
  --train_domains arxiv wikipedia reddit wikihow peerread \
    --train_generators chatGPT davinci cohere dolly flant5 \
  --val_generator flant5 \
  --epochs 3 \
  --batch_size 64 \
  --bf16 \
  --use_wandb \
  --run_name balanced-weighted \
  --output_dir runs/balanced-weighted
```

---

## Consolidated Documentation

All training documentation is now in this file:
- ✅ M4 cleaning (completed: 63k samples, 92% retention)
- ✅ Class imbalance handling (weighted loss recommended)
- ✅ Stratified splitting (preserves 66/34 ratio)
- ✅ Complete training script

**Other MD files to keep:**
- `README.md` - Project overview
- `QUICKSTART.md` - Quick commands
- `TESTING_GUIDE.md` - Model evaluation

**MD files to archive/delete:**
- `M4_CLEANING_GUIDE.md` → consolidated here
- `M4_CLEANING_QUICK_REF.md` → consolidated here
- `RUNPOD_TRAINING.md` → consolidated here

---

## Expected Results

### Training Time (A6000)
- ~6-8 hours for 3 epochs with 186k samples

### Performance Expectations
- **Accuracy**: 93-96%
- **F1 Score**: 92-95%
- **False Positive Rate**: <5% on casual human text

### Benefits of This Approach
✅ Uses all data (no waste)  
✅ Weighted loss handles imbalance  
✅ Stratified split preserves ratio  
✅ Cleaned data reduces bias  
✅ Balanced formal + casual human text  

---

## Summary

**Dataset:** 186k samples (66% human, 34% AI)  
**Solution:** Weighted loss + stratified split  
**Training:** ~6-8 hours on A6000  
**Output:** Balanced model with low false positives  

🚀 Ready to train on RunPod!
