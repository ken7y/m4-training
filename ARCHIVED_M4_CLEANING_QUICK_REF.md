# Quick Reference: M4 Data Cleaning

## TL;DR - What You Need to Know

### Problem
- ❌ **Reddit data**: Cleaned (60k posts) 
- ❌ **M4 data**: NOT cleaned - has HTML, markdown, URLs, formatting
- ❌ **Training inconsistency**: Model learns formatting artifacts instead of AI writing style

### Solution
Clean M4 data with same aggressive cleaning as Reddit before training on RunPod.

---

## Quick Start (3 Commands)

```bash
# 1. Clean M4 dataset
python clean_m4_data.py

# 2. Verify cleaning worked
ls -lh M4_cleaned/data/*.jsonl

# 3. Train with cleaned data
python train.py \
  --data_dir M4_cleaned/data \
  --train_domains arxiv wikipedia reddit wikihow \
  --train_generators chatGPT davinci cohere dolly \
  --val_generator flant5 \
  --epochs 3 \
  --batch_size 64 \
  --bf16
```

**Time**: ~5-10 minutes to clean, 4-6 hours to train on A6000

---

## What Gets Cleaned

| Artifact | Example | After Cleaning |
|----------|---------|----------------|
| HTML entities | `&nbsp;` `&amp;` | ` ` `&` |
| HTML tags | `<p>text</p>` | `text` |
| Markdown links | `[text](url)` | `text` |
| URLs | `http://example.com` | _(removed)_ |
| Markdown bold | `**bold**` | `bold` |
| Code blocks | ` ```code``` ` | _(removed)_ |
| Inline code | `` `code` `` | `code` |
| LaTeX math | `$equation$` | _(removed)_ |

**Same cleaning applied to**:
- ✅ `human_text` field (formal academic)
- ✅ `machine_text` field (AI-generated)

---

## Expected Results

### Before Cleaning
```
Total M4 samples:    ~50,000 lines
Usable:              ~50,000 (raw with artifacts)
```

### After Cleaning
```
Total M4 samples:    ~50,000 lines
✅ Cleaned:           ~47,000 (94% retention)
❌ Removed:           ~3,000 (too short after cleaning)
```

### Combined Training Dataset
```
Reddit (casual human):   60,000 samples
M4 (formal human):      ~23,500 samples (half of cleaned M4)
M4 (AI-generated):      ~23,500 samples (half of cleaned M4)
────────────────────────────────────────
TOTAL:                 ~107,000 samples
```

---

## Key Files

| File | Purpose | Location |
|------|---------|----------|
| `clean_m4_data.py` | Cleaning script | Root directory |
| `M4_CLEANING_GUIDE.md` | Full documentation | Root directory |
| `M4/data/*.jsonl` | Original M4 files | `M4/data/` |
| `M4_cleaned/data/*.jsonl` | **Cleaned M4 files** | `M4_cleaned/data/` |
| `reddit_training_60k.jsonl` | Pre-cleaned Reddit | Root directory |

---

## Training on RunPod

### With Cleaned M4 Data (RECOMMENDED)

```bash
# Upload to RunPod
# /workspace/m4-training/

# Run training
python train.py \
  --data_dir M4_cleaned/data \
  --train_domains arxiv wikipedia reddit wikihow \
  --train_generators chatGPT davinci cohere dolly \
  --val_generator flant5 \
  --epochs 3 \
  --batch_size 64 \
  --bf16 \
  --use_wandb \
  --run_name m4-cleaned-balanced \
  --output_dir runs/m4-cleaned
```

### Benefits of Cleaned Data
✅ Consistent with Reddit cleaning  
✅ No formatting artifacts bias  
✅ Better generalization  
✅ More robust AI detection  
✅ Lower false positive rate  

---

## Verification

### Quick Check (30 seconds)
```bash
# Count cleaned files
ls M4_cleaned/data/*.jsonl | wc -l
# Should show: 32 files

# Check one file
head -1 M4_cleaned/data/arxiv_chatGPT.jsonl | python -m json.tool
# Should be valid JSON without HTML/markdown
```

### Detailed Check (3 minutes)
```bash
# Test with training script
python train.py \
  --data_dir M4_cleaned/data \
  --train_domains arxiv \
  --train_generators chatGPT \
  --val_generator davinci \
  --max_train_samples 200 \
  --max_val_samples 50 \
  --epochs 1 \
  --batch_size 8 \
  --output_dir runs/test-cleaned

# Should run without errors and show:
# "✅ Loaded 400 samples total" (200 x 2 for human+machine)
```

---

## Comparison: Before vs After Training

### Training WITHOUT Cleaning (Current)
```
Dataset: 60k Reddit (cleaned) + 50k M4 (raw)
Problems:
  - M4 has HTML artifacts (&nbsp;, <p>, etc.)
  - M4 has markdown formatting (**bold**, [links])
  - Model learns: "formatted text = AI"
  - High false positive on casual human text
  - Inconsistent data quality
```

### Training WITH Cleaning (Recommended)
```
Dataset: 60k Reddit (cleaned) + 47k M4 (cleaned)
Benefits:
  - Consistent cleaning across all data
  - No formatting artifacts bias
  - Model learns: "AI writing patterns" not "formatting"
  - Lower false positive on casual text
  - Better real-world performance
```

---

## FAQ

### Q: Will cleaning reduce accuracy?
**A**: No. Cleaning **improves** generalization by removing formatting bias.

### Q: How long does cleaning take?
**A**: ~5-10 minutes for entire M4 dataset (32 files, ~50k lines)

### Q: Can I clean Reddit data too?
**A**: Already done! `reddit_training_60k.jsonl` is pre-cleaned.

### Q: What if some files are missing?
**A**: Script automatically skips missing files and reports them.

### Q: Should I clean before or after uploading to RunPod?
**A**: **Before**. Clean locally, then upload cleaned files to RunPod.

---

## Commands Cheatsheet

```bash
# Clean M4 data
python clean_m4_data.py

# Verify output
ls -lh M4_cleaned/data/

# Quick test
python train.py --data_dir M4_cleaned/data --max_train_samples 200 --epochs 1

# Full training (RunPod)
python train.py --data_dir M4_cleaned/data --epochs 3 --batch_size 64 --bf16 --use_wandb

# Compare before/after
diff <(head -1 M4/data/arxiv_chatGPT.jsonl) <(head -1 M4_cleaned/data/arxiv_chatGPT.jsonl)
```

---

## Summary

| Aspect | Status |
|--------|--------|
| Reddit data | ✅ Pre-cleaned (60k) |
| M4 data | ❌ Needs cleaning (50k) |
| Cleaning script | ✅ Ready (`clean_m4_data.py`) |
| Expected output | ✅ ~47k cleaned samples (94% retention) |
| Training ready | ⏳ After cleaning |

**Next step**: Run `python clean_m4_data.py` → Upload to RunPod → Train! 🚀
