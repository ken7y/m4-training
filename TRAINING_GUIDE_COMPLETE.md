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

## Complete Training Script (with Weighted Loss)

Create `train_balanced.py`:

```python
#!/usr/bin/env python3
"""
Balanced Training: M4 (cleaned) + Reddit (60k)
With weighted loss for class imbalance
"""

import json
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import numpy as np

# ============================================================
# 1. LOAD DATA
# ============================================================

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

print("="*80)
print("LOADING DATA")
print("="*80)

# Load Reddit (60k casual human)
print("\n1. Loading Reddit data...")
reddit_data = load_jsonl('reddit_training_60k.jsonl')
print(f"   Reddit posts: {len(reddit_data):,}")

# Load M4 cleaned data (formal human + AI)
print("\n2. Loading M4 cleaned data...")
m4_files = [
    # ArXiv
    'M4_cleaned/data/arxiv_chatGPT.jsonl',
    'M4_cleaned/data/arxiv_davinci.jsonl',
    'M4_cleaned/data/arxiv_cohere.jsonl',
    'M4_cleaned/data/arxiv_dolly.jsonl',
    'M4_cleaned/data/arxiv_flant5.jsonl',
    
    # Wikipedia
    'M4_cleaned/data/wikipedia_chatgpt.jsonl',
    'M4_cleaned/data/wikipedia_davinci.jsonl',
    'M4_cleaned/data/wikipedia_cohere.jsonl',
    'M4_cleaned/data/wikipedia_dolly.jsonl',
    
    # WikiHow
    'M4_cleaned/data/wikihow_chatGPT.jsonl',
    'M4_cleaned/data/wikihow_davinci.jsonl',
    'M4_cleaned/data/wikihow_cohere.jsonl',
    'M4_cleaned/data/wikihow_dolly2.jsonl',
    
    # PeerRead
    'M4_cleaned/data/peerread_chatgpt.jsonl',
    'M4_cleaned/data/peerread_davinci.jsonl',
    'M4_cleaned/data/peerread_cohere.jsonl',
    'M4_cleaned/data/peerread_dolly.jsonl',
    
    # M4 Reddit (different from our 60k)
    'M4_cleaned/data/reddit_chatGPT.jsonl',
    'M4_cleaned/data/reddit_davinci.jsonl',
    'M4_cleaned/data/reddit_dolly.jsonl',
    'M4_cleaned/data/reddit_flant5.jsonl',
]

all_data = []

for filepath in m4_files:
    try:
        m4_lines = load_jsonl(filepath)
        for item in m4_lines:
            # Extract human text
            human_text = item.get('human_text') or item.get('text')
            if human_text:
                all_data.append({
                    'text': human_text,
                    'label': 0,  # human
                    'source': filepath.split('/')[-1]
                })
            
            # Extract machine text
            machine_text = item.get('machine_text')
            if machine_text:
                all_data.append({
                    'text': machine_text,
                    'label': 1,  # AI
                    'source': filepath.split('/')[-1]
                })
        print(f"   ✅ Loaded {filepath.split('/')[-1]}: {len(m4_lines)} lines")
    except Exception as e:
        print(f"   ⚠️  Error loading {filepath}: {e}")

# Add Reddit data
all_data.extend(reddit_data)

print(f"\n{'='*80}")
print(f"DATASET STATISTICS")
print(f"{'='*80}")

# Count by label
human_count = sum(1 for d in all_data if d['label'] == 0)
ai_count = sum(1 for d in all_data if d['label'] == 1)
total_count = len(all_data)

print(f"Human samples: {human_count:,} ({human_count/total_count*100:.1f}%)")
print(f"AI samples:    {ai_count:,} ({ai_count/total_count*100:.1f}%)")
print(f"Total:         {total_count:,}")

# ============================================================
# 2. STRATIFIED SPLIT
# ============================================================

# In the CLI workflow, enable this with `--stratified_split` (adjust holdout via `--validation_split`).

print(f"\n{'='*80}")
print(f"CREATING STRATIFIED SPLIT")
print(f"{'='*80}")

# Extract labels for stratification
labels = [d['label'] for d in all_data]

# 90/10 stratified split
train_data, val_data = train_test_split(
    all_data,
    test_size=0.1,
    stratify=labels,
    random_state=42
)

print(f"\nTrain set: {len(train_data):,}")
print(f"  Human: {sum(1 for d in train_data if d['label']==0):,}")
print(f"  AI:    {sum(1 for d in train_data if d['label']==1):,}")

print(f"\nValidation set: {len(val_data):,}")
print(f"  Human: {sum(1 for d in val_data if d['label']==0):,}")
print(f"  AI:    {sum(1 for d in val_data if d['label']==1):,}")

# Create datasets
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# ============================================================
# 3. TOKENIZATION
# ============================================================

print(f"\n{'='*80}")
print(f"TOKENIZING")
print(f"{'='*80}")

model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=512,
        padding=False  # Will pad in batches
    )

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['text', 'source']
)

val_dataset = val_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['text', 'source']
)

print(f"✅ Tokenization complete")

# ============================================================
# 4. MODEL & WEIGHTED LOSS
# ============================================================

print(f"\n{'='*80}")
print(f"SETTING UP MODEL WITH WEIGHTED LOSS")
print(f"{'='*80}")

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

# Calculate class weights (inverse frequency)
weight_human = total_count / (2 * human_count)
weight_ai = total_count / (2 * ai_count)

class_weights = torch.tensor([weight_human, weight_ai]).cuda()

print(f"\nClass weights (for 66/34 imbalance):")
print(f"  Human (label 0): {weight_human:.3f}")
print(f"  AI (label 1):    {weight_ai:.3f}")
print(f"  → AI loss weighted {weight_ai/weight_human:.2f}x higher (1.95x)")

# Custom weighted loss function
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Weighted cross-entropy loss
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

# ============================================================
# 5. METRICS
# ============================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# ============================================================
# 6. TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(
    output_dir="runs/balanced-m4-reddit",
    num_train_epochs=3,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=500,
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    bf16=True,  # Use bfloat16 on A6000
    report_to="wandb",
    run_name="balanced-m4-reddit-weighted",
)

# ============================================================
# 7. TRAIN
# ============================================================

print(f"\n{'='*80}")
print(f"STARTING TRAINING")
print(f"{'='*80}")

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()

# ============================================================
# 8. SAVE
# ============================================================

print(f"\n{'='*80}")
print(f"SAVING MODEL")
print(f"{'='*80}")

trainer.save_model("final-model-balanced")
tokenizer.save_pretrained("final-model-balanced")

print(f"\n✅ Training complete!")
print(f"📁 Model saved to: final-model-balanced/")
```

---

## Quick Start Commands

### 1. Clean M4 Data (if not done)
```bash
python3 clean_m4_data.py
```

### 2. Train with Weighted Loss
```bash
python3 train_balanced.py
```

### 3. Or use existing train.py with modifications
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
