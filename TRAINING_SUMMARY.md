# Training Summary - English M4 + Reddit (FINAL)

## ✅ Data Cleaned & Ready

### What Was Cleaned
```bash
M4 English files: 28 files → 63,117 lines cleaned (92.1% retention)
✅ arxiv (6 files)
✅ wikipedia (5 files)  
✅ wikihow (5 files)
✅ peerread (6 files)
✅ reddit M4 version (6 files)

❌ EXCLUDED (non-English):
   - arabic_chatGPT.jsonl
   - bulgarian_*.jsonl
   - germanwikipedia_*.jsonl
   - id-newspaper_*.jsonl
   - qazh_*.jsonl
   - russian_*.jsonl
   - urdu_*.jsonl
```

---

## 📊 Final Dataset Composition (English Only)

### Source Breakdown
```
M4 Cleaned (English):
  63,117 lines × 2 texts = 126,234 samples
  ├─ 63,117 human (formal: papers, wiki, etc.)
  └─ 63,117 AI (ChatGPT, Davinci, Cohere, Dolly)

Reddit Training:
  60,000 samples (all human, casual, pre-ChatGPT)

COMBINED TOTAL:
  Human: 123,117 samples (66.1%)
    ├─ 60,000 casual (Reddit)
    └─ 63,117 formal (M4)
  AI:     63,117 samples (33.9%)
    └─ 63,117 formal (M4)
  ─────────────────────────────
  TOTAL: 186,234 samples
```

---

## ⚖️ Class Imbalance Analysis

### Imbalance Ratio
```
Human:AI = 123,117:63,117 = 1.95:1

This is MILD imbalance (66%/34%)
- NOT severe (would be >80%/20%)
- Weighted loss is sufficient
- No need for undersampling/oversampling
```

### Class Weights Calculation
```python
num_human = 123117
num_ai = 63117
total = 186234

# Inverse frequency weighting
weight_human = total / (2 * num_human) = 0.756
weight_ai = total / (2 * num_ai) = 1.476
class_weights = torch.tensor([0.756, 1.476])
```

---

## 🎯 Training Strategy

### ✅ RECOMMENDED: Weighted Loss
**Why:** Uses all data, simple, effective for mild imbalance

Enable it from the CLI with `--use_class_weights` (handled automatically inside `train.py`).

**How it works:**
- Class weights are computed automatically from your training split
- AI class gets ~1.95x higher weight to compensate for fewer samples
- Implemented in `WeightedTrainer` class (see `train.py:30-50`)

### ✅ Stratified Split
Preserve 66/34 ratio in train/val splits using sklearn's `train_test_split` with stratification.

Enable with `--stratified_split` flag. Default 10% validation split can be changed via `--validation_split`.

---

## 📝 Training Command

```bash
python3 train.py \
  --data_dir M4_cleaned/data \
  --reddit_file reddit_training_60k.jsonl \
  --use_class_weights \
  --stratified_split \
  --train_domains arxiv wikipedia reddit wikihow peerread \
  --train_generators chatGPT davinci cohere dolly \
  --val_generator flant5 \
  --epochs 3 \
  --batch_size 64 \
  --bf16 \
  --use_wandb \
  --run_name balanced-english-weighted \
  --output_dir runs/balanced-english
```

**Training Time:** ~6-8 hours on A6000 (186k samples)

---

## 🎓 Expected Performance

### With Weighted Loss
```
Accuracy:  93-96%
F1 Score:  92-95%
Precision: 91-94%
Recall:    93-96%

False Positive Rate: <5% on casual human text
```

### Benefits of This Approach
✅ **All data used** - No waste from undersampling  
✅ **Balanced learning** - AI class gets 1.95x attention  
✅ **Diverse human text** - Formal (M4) + Casual (Reddit)  
✅ **Clean data** - HTML/markdown removed from M4  
✅ **Stratified splits** - Preserves ratio in train/val  

---

## 📋 Checklist Before Training

- [x] M4 data cleaned (63,117 lines, English only)
- [x] Reddit data ready (60,000 samples, pre-cleaned)
- [x] Class weights calculated (0.756, 1.476)
- [x] Weighted loss implemented (WeightedTrainer)
- [x] Stratified split enabled (train_test_split)
- [ ] Upload to RunPod
- [ ] Run training command
- [ ] Monitor W&B dashboard

---

## 🗂️ Files Status

### Ready to Use
```
✅ M4_cleaned/data/          (28 English files, 63,117 lines)
✅ reddit_training_60k.jsonl (60,000 samples)
✅ train.py                  (updated with WeightedTrainer)
✅ TRAINING_GUIDE_COMPLETE.md (full documentation)
```

### For Reference
```
📖 README.md               (project overview)
📖 QUICKSTART.md          (quick commands)
📖 TESTING_GUIDE.md       (model evaluation)
```

### Can Archive/Delete
```
🗑️ M4_CLEANING_GUIDE.md     (consolidated)
🗑️ M4_CLEANING_QUICK_REF.md (consolidated)
🗑️ RUNPOD_TRAINING.md       (consolidated)
```

---

## 🚀 Ready to Train!

**Dataset:** 186,234 English samples (66% human, 34% AI)  
**Strategy:** Weighted loss (1.95x for AI) + stratified split  
**Time:** 6-8 hours on RunPod A6000  
**Cost:** ~$2-3 on RunPod  

All data is cleaned, weighted, and ready for balanced training! 🎉
