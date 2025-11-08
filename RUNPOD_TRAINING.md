# RunPod Training Guide - M4 + Reddit AI Detector

## Overview
This guide explains how to train a new RoBERTa-based AI text detection model using a balanced dataset combining formal academic text (M4) and casual Reddit posts.

## Problem Statement
The original M4 model (F1: 97.12%) was trained primarily on formal text (arXiv papers, Wikipedia). This caused high false positive rates (19.4%) on casual human writing like Reddit posts, as the model learned "human = formal academic writing".

## Solution
Train a new model with balanced data:
- **~45k formal human text** (M4: arXiv, Wikipedia)
- **60k casual human text** (Reddit posts from 2019, pre-ChatGPT)
- **~45k AI-generated text** (M4: ChatGPT, Davinci, Cohere, Dolly, etc.)
- **Total: ~150k balanced samples**

---

## Data Preparation

### Training Data Files
1. **reddit_training_60k.jsonl** (48MB)
   - 60,000 cleaned Reddit posts from 2019
   - Pre-ChatGPT era (before Nov 30, 2022)
   - All posts >= 100 chars after cleaning
   - Format: `{"text": "...", "label": 0, "source": "reddit_...", "id": "..."}`

2. **M4 Dataset** (in `M4/data/`)
   - Multiple JSONL files for different sources
   - Human sources: arXiv, Wikipedia, PeerRead, WikiHow
   - AI sources: ChatGPT, Davinci, Cohere, Dolly, Bloomz, etc.

### Data Format
```json
{"text": "The post content here...", "label": 0, "source": "reddit_AskReddit", "id": "abc123"}
```
- `label`: 0 = Human, 1 = AI
- `source`: Dataset origin (reddit_*, arxiv_*, etc.)
- `id`: Unique identifier

### Data Cleaning Applied
All Reddit data has been aggressively cleaned to remove:
- HTML tags and entities (including nested: &amp;amp; → &amp; → &)
- Markdown formatting ([text](url), **bold**, etc.)
- URLs (http://, www., .com domains)
- Code blocks and inline code
- Excessive whitespace and newlines

---

## Training Script

Create `train_balanced_model.py`:

```python
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ============================================================
# LOAD DATA
# ============================================================
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

print("Loading training data...")

# Reddit human posts (casual)
reddit_data = load_jsonl('reddit_training_60k.jsonl')
print(f"Reddit posts: {len(reddit_data):,}")

# M4 human posts (formal) - filter English only
m4_human = []
human_sources = ['arxiv_', 'wikipedia_', 'peerread_', 'wikihow_', 'reddit_']
for source in human_sources:
    # Load corresponding M4 files - you'll need to adapt this
    # based on which M4 files contain human text with label=0
    pass

# M4 AI posts
m4_ai = []
ai_models = ['chatgpt', 'davinci', 'cohere', 'dolly', 'bloomz']
for model in ai_models:
    # Load M4 AI files with label=1
    pass

# Combine
all_data = reddit_data + m4_human + m4_ai
print(f"Total samples: {len(all_data):,}")

# Count labels
human_count = sum(1 for d in all_data if d['label'] == 0)
ai_count = sum(1 for d in all_data if d['label'] == 1)
print(f"Human samples: {human_count:,}")
print(f"AI samples: {ai_count:,}")

# ============================================================
# PREPARE DATASET
# ============================================================
texts = [d['text'] for d in all_data]
labels = [d['label'] for d in all_data]

dataset = Dataset.from_dict({
    'text': texts,
    'label': labels
})

# Train/test split
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset['train']
test_dataset = dataset['test']

print(f"\nTrain samples: {len(train_dataset):,}")
print(f"Test samples: {len(test_dataset):,}")

# ============================================================
# TOKENIZATION
# ============================================================
model_name = 'roberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=512,
        padding=False
    )

print("\nTokenizing datasets...")
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ============================================================
# MODEL
# ============================================================
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

# ============================================================
# METRICS
# ============================================================
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

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
# TRAINING
# ============================================================
training_args = TrainingArguments(
    output_dir='./balanced-model-output',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy='steps',
    eval_steps=500,
    save_strategy='steps',
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    fp16=True,  # Use mixed precision for faster training
    dataloader_num_workers=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("\nStarting training...")
trainer.train()

# ============================================================
# SAVE MODEL
# ============================================================
print("\nSaving final model...")
trainer.save_model('./final-balanced-model')
tokenizer.save_pretrained('./final-balanced-model')

print("\n✅ Training complete!")
```

---

## RunPod Setup

### 1. Create RunPod Instance
- Template: PyTorch (with transformers pre-installed)
- GPU: A100 or RTX 4090 recommended
- Disk: 100GB minimum
- Image: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel`

### 2. Install Dependencies
```bash
pip install transformers datasets torch scikit-learn
```

### 3. Upload Data
Upload these files to RunPod:
```bash
rsync -avz reddit_training_60k.jsonl runpod:/workspace/
rsync -avz M4/ runpod:/workspace/M4/
rsync -avz train_balanced_model.py runpod:/workspace/
```

Or use RunPod's file upload feature.

### 4. Run Training
```bash
cd /workspace
python3 train_balanced_model.py
```

### 5. Monitor Training
Training will take 2-4 hours depending on GPU. Logs will show:
- Training loss
- Evaluation metrics (accuracy, precision, recall, F1)
- Checkpoints saved every 500 steps

### 6. Download Model
Once complete, download the trained model:
```bash
rsync -avz runpod:/workspace/final-balanced-model/ ./final-balanced-model/
```

---

## Expected Results

With balanced training data, the model should achieve:
- **Overall accuracy**: 95%+ on test set
- **Human detection**: 90%+ (improved from 51% on casual text)
- **AI detection**: 95%+ (maintained high performance)
- **False positives on Reddit**: <5% (down from 19.4%)

The model will learn that humans write both formally (academic) and casually (Reddit), reducing false positives on informal text.

---

## Testing the Trained Model

After training, test on multiple datasets:

```python
# Test on formal text (arXiv)
# Test on casual text (Reddit WSB)
# Test on balanced mixed dataset
# Test on Kaggle AI detection dataset
```

Use the existing test scripts:
- `test_accuracy.py` - Test on balanced dataset
- `test_wsb_human_detection.py` - Test on Reddit posts
- `test_all_models_kaggle.py` - Test on Kaggle data

---

## Data Sources

### Reddit Data (60k samples)
- **Source**: reddit-bigset dataset (1M+ posts from 2019)
- **Filter**: Posts from 2019 (pre-ChatGPT)
- **Processing**: `process_full_reddit_bigset.py` → cleaned 1M posts → sampled 60k
- **Quality**: All >= 100 chars after aggressive cleaning

### M4 Data (~90k samples)
- **Human sources**: arXiv, Wikipedia, PeerRead, WikiHow, Reddit
- **AI sources**: ChatGPT, Davinci, Cohere, Dolly, Bloomz, FlanT5, LLaMA
- **Languages**: English only (filter out German, Bulgarian, Arabic, etc.)

---

## Files Included

### Training Data
- `reddit_training_60k.jsonl` - 60k Reddit posts (48MB)
- `M4/data/*.jsonl` - M4 dataset files

### Data Processing Scripts
- `process_full_reddit_bigset.py` - Process 1M Reddit posts
- `sample_reddit_bigset.py` - Sample from large dataset
- `clean_reddit_data.py` - Clean Reddit posts
- `clean_reddit_bigset.py` - Clean bigset sample
- `prepare_reddit_training_data.py` - Prepare WSB data

### Testing Scripts
- `test_accuracy.py` - Test model accuracy
- `test_wsb_human_detection.py` - Test on Reddit posts
- `test_all_models_kaggle.py` - Compare models on Kaggle data
- `detector.py` - Streamlit web interface

---

## Notes

### Why Reddit?
- Reddit posts are authentic human writing from pre-ChatGPT era
- Casual, informal language (opposite of academic papers)
- Diverse topics and writing styles
- Balances the formal bias in M4 dataset

### Data Balance Rationale
- **60k Reddit** (casual human) + **~45k M4** (formal human) = **~105k human**
- **~45k M4 AI** = balanced 2:1 human:AI ratio
- Prevents overfitting to one writing style
- Teaches model that humans write both formally and casually

### Important Considerations
1. **English only**: Filter out non-English M4 files (German, Bulgarian, Arabic, etc.)
2. **Pre-ChatGPT data**: Reddit from 2019, M4 from before Nov 2022
3. **Cleaning**: All Reddit data aggressively cleaned (no HTML/markdown/URLs)
4. **Balance**: ~2:1 human:AI ratio is typical for detection tasks

---

## Troubleshooting

**Out of memory?**
- Reduce `per_device_train_batch_size` to 8
- Reduce `max_length` to 256

**Training too slow?**
- Use A100 GPU instead of RTX 4090
- Enable `fp16=True` (mixed precision)
- Increase `dataloader_num_workers`

**Low accuracy?**
- Check data balance (human vs AI)
- Verify labels are correct (0=human, 1=AI)
- Increase training epochs to 5

**Model bias persists?**
- Add more Reddit data (increase to 80k)
- Add more diverse casual sources (Twitter, forums)
- Check if M4 human data includes casual text

---

## Contact

For questions or issues, check:
- `TESTING_GUIDE.md` - Model testing instructions
- `M4/README.md` - M4 dataset documentation
