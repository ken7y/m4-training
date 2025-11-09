# M4 AI Text Detection Training

**Train a RoBERTa model to detect AI-generated text with 92-95% F1 score.**

Simple, production-ready training on RunPod A6000 (6-8 hours, ~$10-15).

---

## 🚀 Quick Start on RunPod

### 1. Launch RunPod Instance
- **GPU:** A6000 (48GB VRAM)
- **Template:** PyTorch 2.0+ with CUDA 11.8+
- **Disk:** 50GB minimum

### 2. Clone & Upload Data
```bash
# On RunPod terminal
git clone <your-repo-url>
cd m4-training

# Upload your cleaned data (M4_cleaned/ and reddit_training_60k.jsonl)
# Use RunPod's file upload or scp
```

### 3. Run Setup
```bash
./runpod_setup.sh
```

This will:
- ✅ Install dependencies
- ✅ Verify data exists (M4_cleaned/data + reddit_training_60k.jsonl)
- ✅ Run quick validation test

### 4. Start Training
```bash
./start_training.sh
```

That's it! Training will run for 6-8 hours.

**Optional:** Login to W&B first for tracking: `wandb login`

---

## 📊 What You're Training

### Dataset (186k samples)
- **M4 Cleaned:** 63,117 lines → 126,234 samples
  - 63k human (formal: papers, Wikipedia, wikihow)
  - 63k AI (ChatGPT, Davinci, Cohere, Dolly, Flan-T5)
- **Reddit:** 60,000 human samples (casual text, pre-ChatGPT)

**Total:** 123k human (66%) + 63k AI (34%)

### Strategy
- **Weighted loss:** AI class gets 1.95x weight to compensate for imbalance
- **Stratified split:** 90% train, 10% validation (preserves 66/34 ratio)
- **Chunking:** Long texts split into 512-token chunks with 50-token overlap
- **Preprocessing:** Digit normalization, HTML/markdown removal

### Expected Results
- **Accuracy:** 93-96%
- **F1 Score:** 92-95%
- **False Positive Rate:** <5% on casual human text

---

## 📁 Project Structure

```
m4-training/
├── runpod_setup.sh          # Run this first
├── start_training.sh         # Then run this to train
├── train.py                  # Main training script
├── predict.py                # Inference script
├── requirements.txt          # Dependencies
│
├── M4_cleaned/data/          # Upload: 28 cleaned M4 files (63k lines)
├── reddit_training_60k.jsonl # Upload: Reddit data (60k lines)
│
└── runs/                     # Training output (created automatically)
    └── balanced-m4-reddit/   # Your trained model
```

---

## 🛠️ Advanced Usage

### Custom Training Command
If you want to modify settings, edit `start_training.sh` or run directly:

```bash
python3 train.py \
  --data_dir M4_cleaned/data \
  --reddit_file reddit_training_60k.jsonl \
  --use_class_weights \
  --stratified_split \
  --validation_split 0.1 \
  --train_domains arxiv wikipedia reddit wikihow peerread \
  --train_generators chatGPT davinci cohere dolly flant5 \
  --epochs 3 \
  --batch_size 64 \
  --lr 2e-5 \
  --dropout 0.2 \
  --bf16 \
  --use_wandb \
  --run_name my-custom-run \
  --output_dir runs/my-model
```

### Available Options
```bash
# Data
--data_dir              # Path to M4_cleaned/data
--reddit_file           # Path to reddit_training_60k.jsonl
--train_domains         # arxiv wikipedia reddit wikihow peerread
--train_generators      # chatGPT davinci cohere dolly flant5

# Training
--epochs                # Number of epochs (default: 3)
--batch_size            # Batch size (default: 64 for A6000)
--lr                    # Learning rate (default: 2e-5)
--dropout               # Dropout rate (default: 0.2)

# Strategy
--use_class_weights     # Enable weighted loss for imbalance
--stratified_split      # Enable stratified train/val split
--validation_split      # Val split ratio (default: 0.1)

# Performance
--bf16                  # Use bfloat16 (A6000 recommended)
--fp16                  # Use float16 (other GPUs)

# Tracking
--use_wandb             # Enable W&B logging
--run_name              # W&B run name
```

---

## 🧪 Testing Your Model

### Run Inference
```bash
python3 predict.py \
  --model_path runs/balanced-m4-reddit \
  --input_file test_data.jsonl \
  --output_file predictions.jsonl
```

### Input Format (JSONL)
```json
{"text": "Your text to analyze..."}
{"text": "Another sample..."}
```

### Output Format
```json
{
  "prediction": 0,           // 0=human, 1=AI
  "label": "human",
  "confidence": 0.94,
  "prob_human": 0.94,
  "prob_machine": 0.06,
  "num_chunks": 2
}
```

---

## 📚 Dataset Preparation

If you need to clean the M4 data yourself:

```bash
# 1. Clone M4 dataset
git clone https://github.com/mbzuai-nlp/M4.git

# 2. Clean the data (removes HTML, markdown, URLs)
python3 clean_m4_data.py
# Creates: M4_cleaned/data/ with 28 files, ~63k lines

# 3. Prepare Reddit data (if needed)
python3 prepare_reddit_training_data.py
# Creates: reddit_training_60k.jsonl
```

---

## ⚙️ Configuration Details

### Why These Settings?

**Batch Size 64:**
- A6000 has 48GB VRAM
- RoBERTa-base uses ~18GB with batch 64 (37% utilization)
- Faster training with underutilized GPU

**Learning Rate 2e-5:**
- Standard for RoBERTa fine-tuning
- Proven optimal for full model training

**Weighted Loss:**
- Handles 66/34 class imbalance
- AI class weighted 1.95x higher
- No data wasted (vs undersampling)

**Stratified Split:**
- Preserves class ratio in train/val
- More reliable validation metrics

**BF16 Precision:**
- A6000 native support
- Better stability than FP16
- Same speed, better quality

---

## 🐛 Troubleshooting

### "No training data found"
- Check `M4_cleaned/data/` directory exists
- Verify files are .jsonl format
- Ensure at least 28 files present

### "Reddit file not found"
- Check `reddit_training_60k.jsonl` is in project root
- Verify file has 60,000 lines

### "OOM Error" (Out of Memory)
```bash
# Reduce batch size
./start_training.sh --batch_size 32  # or 16
```

### "Model not learning" (F1 stuck at ~50%)
- Verify data loaded correctly (should see ~186k total samples)
- Check label distribution (~66% human, 34% AI)
- Ensure `--use_class_weights` flag is set

### GPU not utilized
- Verify `--bf16` flag is set
- Check GPU with: `watch -n 1 nvidia-smi`
- Should see 95-100% utilization

---

## 📈 Monitoring Training

### With W&B (Recommended)
```bash
wandb login  # Run once
./start_training.sh
# Visit: https://wandb.ai/your-username
```

**What's tracked:**
- ✅ Loss (training & validation)
- ✅ F1, Accuracy, Precision, Recall
- ✅ Learning rate schedule
- ✅ GPU utilization
- ❌ Model checkpoints (disabled to save space)

### Without W&B
Watch the console output:
- Loss should decrease: 0.3 → 0.1
- eval_f1 should increase: 0.85 → 0.95
- Plateaus after ~2-3 epochs (good!)

---

## 💾 After Training

### Download Your Model
```bash
# From local machine
scp -r runpod:/workspace/m4-training/runs/balanced-m4-reddit ./my-model/
```

### Files You'll Get
```
balanced-m4-reddit/
├── pytorch_model.bin          # Model weights (~480MB)
├── config.json                # Model architecture
├── training_config.json       # Your hyperparameters + metrics
├── tokenizer_config.json      # Tokenizer settings
├── vocab.json                 # Vocabulary
└── merges.txt                 # BPE merges
```

### Use Your Model
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("./my-model")
tokenizer = AutoTokenizer.from_pretrained("./my-model")

# Predict
text = "Your text here..."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
outputs = model(**inputs)
prediction = outputs.logits.argmax(-1).item()  # 0=human, 1=AI
```

---

## 📖 Additional Documentation

For detailed explanations, see:
- `TRAINING_SUMMARY.md` - Dataset composition & strategy details
- `TRAINING_GUIDE_COMPLETE.md` - Full training guide with examples
- `TESTING_GUIDE.md` - Model evaluation guide

---

## 🎯 Key Takeaways

1. **Simple workflow:** Two scripts to run (`runpod_setup.sh` → `start_training.sh`)
2. **Balanced dataset:** 66% human (formal + casual) + 34% AI
3. **Smart strategy:** Weighted loss handles imbalance without wasting data
4. **Production-ready:** Clean code, proper validation, efficient training
5. **6-8 hours:** On A6000 (~$10-15 on RunPod)
6. **92-95% F1:** Expected performance on mixed formal/casual text

---

## ⚡ One-Line Summary

**Upload data → `./runpod_setup.sh` → `./start_training.sh` → Wait 6-8 hours → 95% F1 model** 🚀
