# RunPod Training Checklist

**Use this checklist to ensure smooth training on RunPod.**

---

## ✅ Before Launching RunPod

- [ ] You have `M4_cleaned/data/` directory with 28 .jsonl files (~63k lines total)
- [ ] You have `reddit_training_60k.jsonl` (60,000 lines)
- [ ] Both datasets are cleaned (no HTML, markdown, URLs)
- [ ] You have a W&B account (optional, for tracking)
- [ ] You have your repo URL ready

**If you need to prepare data:**
```bash
# Clean M4 data
python3 clean_m4_data.py

# Verify Reddit data exists
wc -l reddit_training_60k.jsonl  # Should show 60000
```

---

## ✅ RunPod Setup

### 1. Launch Instance
- [ ] GPU: **A6000** (48GB VRAM recommended)
- [ ] Template: PyTorch 2.0+ with CUDA 11.8+
- [ ] Disk: 50GB minimum
- [ ] Start instance

### 2. Connect
- [ ] Click "Connect" → "Start Web Terminal" or SSH

### 3. Clone Repo
```bash
cd /workspace
git clone <your-repo-url>
cd m4-training
```

### 4. Upload Data
**Option A: SCP from local machine**
```bash
# From your local machine
scp -r M4_cleaned/ runpod:/workspace/m4-training/
scp reddit_training_60k.jsonl runpod:/workspace/m4-training/
```

**Option B: RunPod File Upload**
- Use RunPod web UI to upload files
- Upload to `/workspace/m4-training/`

**Verify upload:**
```bash
ls -lh M4_cleaned/data/*.jsonl | wc -l  # Should show 28
wc -l reddit_training_60k.jsonl         # Should show 60000
```

---

## ✅ Training Setup

### 5. Run Setup Script
```bash
./runpod_setup.sh
```

**What it does:**
- Installs dependencies
- Verifies data exists
- Runs quick validation test

**Expected output:**
```
✅ M4 cleaned data: 28 files, 63117 lines
✅ Reddit data: 60000 lines
✅ M4 data loads correctly: 10 samples
✅ Reddit data loads correctly: 10 samples
✅ SETUP COMPLETE!
```

### 6. (Optional) Login to W&B
```bash
wandb login
# Paste your API key from https://wandb.ai/authorize
```

Skip this if you don't want experiment tracking.

---

## ✅ Start Training

### 7. Run Training
```bash
./start_training.sh
```

**Expected timeline:**
- Setup: 3-5 minutes (loading + tokenizing data)
- Training: 6-8 hours (3 epochs with 186k samples)
- Total: ~6-8 hours

### 8. Monitor Training

**Option A: W&B Dashboard**
- Visit https://wandb.ai/your-username
- Watch F1 score increase: 0.85 → 0.95

**Option B: Console**
```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Should see:
# - GPU Util: 95-100%
# - Memory: ~18GB / 48GB
# - Power: 230-280W
```

**Option C: Log file**
```bash
# Training logs are shown in terminal
# Look for:
# - Loss decreasing: 0.3 → 0.1
# - eval_f1 increasing: 0.85 → 0.95
```

---

## ✅ Training Health Checks

### What Good Training Looks Like:
- ✅ Loss decreases steadily
- ✅ eval_f1 increases to 0.92-0.95
- ✅ GPU utilization 95-100%
- ✅ No OOM errors
- ✅ Training completes all 3 epochs

### Red Flags:
- 🚨 eval_f1 stuck at ~0.50 (random chance = not learning)
- 🚨 OOM errors (reduce batch size)
- 🚨 GPU util <50% (check --bf16 flag)
- 🚨 Loss not decreasing

---

## ✅ After Training

### 9. Verify Output
```bash
ls -lh runs/balanced-m4-reddit/

# Should see:
# - pytorch_model.bin (~480MB)
# - config.json
# - training_config.json
# - tokenizer files
```

### 10. Check Final Metrics
```bash
cat runs/balanced-m4-reddit/training_config.json | grep -A 5 final_metrics

# Expected:
# "eval_f1": 0.92-0.95
# "eval_accuracy": 0.93-0.96
```

### 11. Download Model
**From your local machine:**
```bash
scp -r runpod:/workspace/m4-training/runs/balanced-m4-reddit ./my-model/
```

### 12. Test Model (Optional)
```bash
# On RunPod or local machine
python3 predict.py \
  --model_path runs/balanced-m4-reddit \
  --input_file test_data.jsonl \
  --output_file predictions.jsonl
```

---

## ✅ Cleanup

### 13. Stop RunPod Instance
- [ ] Downloaded model
- [ ] Downloaded training_config.json
- [ ] Downloaded any logs you need
- [ ] Terminate RunPod instance

**Cost estimate:**
- A6000: ~$0.70-$0.80/hour
- 8 hours training: ~$5.60-$6.40
- Total with setup: ~$10-15

---

## 🐛 Common Issues & Fixes

### "No training data found"
```bash
# Check files exist
ls M4_cleaned/data/*.jsonl | wc -l  # Should be 28
ls reddit_training_60k.jsonl        # Should exist

# Re-upload if missing
```

### "OOM Error"
```bash
# Edit start_training.sh, change:
--batch_size 64  →  --batch_size 32
```

### "Dataset too small"
```bash
# Verify data loaded
wc -l M4_cleaned/data/*.jsonl | tail -1  # ~63k total
wc -l reddit_training_60k.jsonl          # 60k
```

### "Training stuck / not learning"
```bash
# Check that class weights are enabled
grep "use_class_weights" start_training.sh  # Should see --use_class_weights
```

### "GPU not utilized"
```bash
# Check bf16 is enabled
grep "bf16" start_training.sh  # Should see --bf16
```

---

## 📋 Quick Reference

### File Locations
```
/workspace/m4-training/          # Your repo
├── M4_cleaned/data/             # M4 dataset (upload)
├── reddit_training_60k.jsonl    # Reddit data (upload)
├── runpod_setup.sh              # Run first
├── start_training.sh            # Run second
└── runs/balanced-m4-reddit/     # Output model
```

### Commands
```bash
# Setup
./runpod_setup.sh

# Train
./start_training.sh

# Monitor GPU
watch -n 1 nvidia-smi

# Check progress
tail -f wandb/latest-run/run-*.log  # If using W&B

# Download model (from local)
scp -r runpod:/workspace/m4-training/runs/balanced-m4-reddit ./
```

---

## ✅ Success Criteria

**You're done when:**
- ✅ Training completed 3 epochs
- ✅ Final F1 score: 0.92-0.95
- ✅ Model saved to `runs/balanced-m4-reddit/`
- ✅ Downloaded model to local machine
- ✅ RunPod instance terminated

**Next steps:**
- Test your model with `predict.py`
- Deploy to production
- Celebrate! 🎉
