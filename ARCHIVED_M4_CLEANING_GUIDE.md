# M4 Dataset Cleaning Guide

## Overview

This guide explains how to clean the M4 dataset to remove HTML, markdown, URLs, and formatting artifacts before training. The cleaning process ensures consistency between the formal M4 data and the casual Reddit data.

## Why Clean M4 Data?

### Current Status
- **Reddit data**: Already cleaned (`reddit_training_60k.jsonl` - 60k cleaned posts)
- **M4 data**: **NOT CLEANED** - contains raw academic text with formatting

### Problems with Uncleaned M4 Data
1. **HTML artifacts**: `&nbsp;`, `&amp;`, `<p>`, etc. from web scraping
2. **Markdown formatting**: `**bold**`, `[links](url)`, code blocks
3. **URLs**: `http://`, `www.`, `.com` domains
4. **LaTeX/Math**: `$equation$` symbols
5. **Inconsistent with Reddit**: Different text characteristics

### Impact on Training
- Model may learn to detect **formatting** instead of **AI writing style**
- Biased towards formatted text = AI, plain text = Human
- Lower generalization to real-world text
- Inconsistent with your cleaned Reddit data

---

## M4 Dataset Structure

### File Format
Each M4 JSONL file contains paired human/AI text:
```json
{
  "prompt": "Generate a 150-220-word abstract...",
  "human_text": "Original academic text...",
  "machine_text": "AI-generated version...",
  "model": "gpt-3.5-turbo",
  "source": "arxiv",
  "source_ID": 704.0007
}
```

### English-Only Files (32 files)

**ArXiv (6 files)** - Scientific papers
- `arxiv_bloomz.jsonl`, `arxiv_chatGPT.jsonl`, `arxiv_cohere.jsonl`
- `arxiv_davinci.jsonl`, `arxiv_dolly.jsonl`, `arxiv_flant5.jsonl`

**Wikipedia (5 files)** - Encyclopedia articles
- `wikipedia_bloomz.jsonl`, `wikipedia_chatgpt.jsonl`, `wikipedia_cohere.jsonl`
- `wikipedia_davinci.jsonl`, `wikipedia_dolly.jsonl`

**WikiHow (5 files)** - How-to guides
- `wikihow_bloomz.jsonl`, `wikihow_chatGPT.jsonl`, `wikihow_cohere.jsonl`
- `wikihow_davinci.jsonl`, `wikihow_dolly2.jsonl`

**PeerRead (6 files)** - Academic peer reviews
- `peerread_bloomz.jsonl`, `peerread_chatgpt.jsonl`, `peerread_cohere.jsonl`
- `peerread_davinci.jsonl`, `peerread_dolly.jsonl`, `peerread_llama.jsonl`

**Reddit (6 files)** - Social media posts
- `reddit_bloomz.jsonl`, `reddit_chatGPT.jsonl`, `reddit_cohere.jsonl`
- `reddit_davinci.jsonl`, `reddit_dolly.jsonl`, `reddit_flant5.jsonl`

**Total**: ~28-32 English files (some may be missing)

---

## Cleaning Process

### What Gets Cleaned

The `clean_m4_data.py` script applies the **same aggressive cleaning** used for Reddit:

1. **HTML entities** (nested): `&amp;amp;` → `&amp;` → `&`
2. **HTML tags**: `<p>`, `<br>`, `<div>`, etc.
3. **Markdown links**: `[text](url)` → `text`
4. **URLs**: Remove `http://`, `www.`, `.com` domains
5. **Markdown formatting**: `**bold**`, `*italic*`, `__underline__`
6. **Code blocks**: ` ```code``` ` and `` `inline` ``
7. **LaTeX math**: `$equation$`
8. **Headers**: `# Title`, `## Section`
9. **Whitespace**: Excessive spaces and newlines

### Quality Control

- **Both texts must pass**: If either `human_text` OR `machine_text` becomes < 100 chars, the entire line is removed
- **Preserves pairing**: Maintains the correspondence between human and AI text
- **No data leakage**: All cleaning is format-based, not content-based

---

## Usage

### Step 1: Clean M4 Data

```bash
cd /path/to/m4-training

# Run cleaning script
python clean_m4_data.py
```

**Expected Output:**
```
================================================================================
CLEANING M4 DATASET
================================================================================

📁 Input:  M4/data
📁 Output: M4_cleaned/data
📏 Min length: 100 characters

================================================================================
Processing: arxiv_chatGPT.jsonl
================================================================================

Results:
  Total lines:          3,000
  ✅ Cleaned:            2,897
  ❌ Skipped (too short): 103
  Retention rate:       96.6%

  Saved to: M4_cleaned/data/arxiv_chatGPT.jsonl

[... processes all 32 files ...]

================================================================================
CLEANING COMPLETE - SUMMARY
================================================================================
Files processed:      32
Files skipped:        0
Total lines:          ~50,000
✅ Total cleaned:      ~47,000
❌ Total skipped:      ~3,000
Overall retention:    94.0%

📁 Cleaned files saved to: M4_cleaned/data/
```

### Step 2: Verify Cleaning

```bash
# Compare before/after
head -1 M4/data/arxiv_chatGPT.jsonl | python -m json.tool
head -1 M4_cleaned/data/arxiv_chatGPT.jsonl | python -m json.tool

# Count lines
wc -l M4/data/*.jsonl
wc -l M4_cleaned/data/*.jsonl
```

### Step 3: Update Training Script

**Option A: Temporary (command-line)**
```bash
python train.py \
  --data_dir M4_cleaned/data \
  --train_domains arxiv wikipedia reddit wikihow \
  --train_generators chatGPT davinci cohere dolly \
  --val_generator flant5 \
  --epochs 3 \
  --batch_size 64 \
  --bf16 \
  --use_wandb \
  --run_name m4-cleaned \
  --output_dir runs/m4-cleaned
```

**Option B: Permanent (edit train.py)**
```python
# Line 45 in train.py
parser.add_argument("--data_dir", type=str, default="M4_cleaned/data", 
                    help="Path to M4 data directory")
```

---

## Training with Cleaned Data

### Balanced Training (Reddit + Cleaned M4)

```python
# In your training script or RUNPOD_TRAINING.md

# 1. Load cleaned Reddit (60k casual human)
reddit_data = load_jsonl('reddit_training_60k.jsonl')
print(f"Reddit posts: {len(reddit_data):,}")

# 2. Load cleaned M4 (formal human + AI)
m4_human = []
m4_ai = []

human_files = [
    'M4_cleaned/data/arxiv_chatGPT.jsonl',  # Use human_text field
    'M4_cleaned/data/wikipedia_chatgpt.jsonl',
    'M4_cleaned/data/peerread_chatgpt.jsonl',
    # ... etc
]

ai_files = [
    'M4_cleaned/data/arxiv_chatGPT.jsonl',  # Use machine_text field
    'M4_cleaned/data/arxiv_davinci.jsonl',
    'M4_cleaned/data/arxiv_cohere.jsonl',
    # ... etc
]

# 3. Combine for balanced dataset
all_data = reddit_data + m4_human + m4_ai
# Expected: ~150k balanced samples
```

### Expected Dataset Sizes (After Cleaning)

| Source | Type | Original | Cleaned | Retention |
|--------|------|----------|---------|-----------|
| Reddit | Human (casual) | 60,000 | 60,000 | 100% (pre-cleaned) |
| M4 ArXiv | Human (formal) | ~18,000 | ~17,000 | ~94% |
| M4 Wikipedia | Human (formal) | ~15,000 | ~14,000 | ~93% |
| M4 All | AI-generated | ~50,000 | ~47,000 | ~94% |
| **TOTAL** | **Balanced** | **~143k** | **~138k** | **~96.5%** |

---

## Cleaning Script Details

### File: `clean_m4_data.py`

**Key Features:**
- ✅ Handles all M4 format variations (standard, Reddit, PeerRead)
- ✅ Cleans both `human_text` and `machine_text` fields
- ✅ Same cleaning function as Reddit (`deep_clean_text`)
- ✅ Quality control: removes if either text < 100 chars
- ✅ Detailed statistics per file
- ✅ Creates new directory to preserve originals

**Configuration:**
```python
input_dir = "M4/data"           # Original files
output_dir = "M4_cleaned/data"   # Cleaned files
min_length = 100                 # Same as Reddit
```

---

## Before/After Examples

### Example 1: HTML Entities
**Before:**
```
"We investigate the continuum limit&nbsp;of polymer quantum mechanics&hellip;"
```
**After:**
```
"We investigate the continuum limit of polymer quantum mechanics..."
```

### Example 2: Markdown Formatting
**Before:**
```
"Results show **significant improvement** in [accuracy](http://example.com)."
```
**After:**
```
"Results show significant improvement in accuracy."
```

### Example 3: Code Blocks
**Before:**
```
"We use `python` to implement ```def model(): pass``` for training."
```
**After:**
```
"We use python to implement for training."
```

---

## Verification

### Check File Integrity
```bash
# Ensure cleaned files have valid JSON
for f in M4_cleaned/data/*.jsonl; do
  echo "Checking $f"
  python -c "import json; [json.loads(line) for line in open('$f')]"
done
```

### Compare Statistics
```bash
# Original
python audit_all_data.py --data_dir M4/data

# Cleaned
python audit_all_data.py --data_dir M4_cleaned/data
```

### Test with Training Script
```bash
# Quick smoke test with cleaned data
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
```

---

## Troubleshooting

### Issue: "File not found"
**Cause**: Some M4 files may not exist in your directory
**Solution**: Script automatically skips missing files and reports them

### Issue: Low retention rate (< 85%)
**Cause**: Too aggressive cleaning or short source texts
**Solution**: 
1. Check `min_length` parameter (default: 100)
2. Lower to 50 for shorter texts: `min_length=50`
3. Inspect removed samples manually

### Issue: Training data imbalance
**Cause**: Different retention rates across domains
**Solution**: Use `--max_train_samples` to balance during training

---

## Best Practices

1. **Always clean before training**: Don't mix cleaned and uncleaned data
2. **Keep originals**: Script creates new directory `M4_cleaned/`
3. **Verify retention**: Check that retention rate is > 90%
4. **Test first**: Run smoke test before full training
5. **Document changes**: Note cleaned data in W&B run name

---

## Summary

✅ **Reddit data**: Already cleaned (60k posts)  
✅ **M4 data**: **MUST BE CLEANED** before training  
✅ **Cleaning script**: `clean_m4_data.py` (same as Reddit cleaning)  
✅ **Output**: `M4_cleaned/data/` (32 cleaned files)  
✅ **Expected**: ~138k balanced samples after cleaning  

**Next**: Train balanced model on RunPod with cleaned M4 + Reddit data!
