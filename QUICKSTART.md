# Quick Start - M4 Training (English Only)

**Just the commands you need.**

---

## Setup (Once)

```bash
git clone <your-repo>
cd m4-training
pip install -r requirements.txt
wandb login  # Optional, for experiment tracking
```

---

## Verify (30 seconds)

```bash
python3 test_fixed_loader.py
# Must show: ✅ ALL TESTS PASSED!
```

---

## Train (4-6 hours)

### Recommended - English Only (95% F1):
```bash
python train.py \
  --model roberta-base \
  --lr 2e-5 \
  --dropout 0.2 \
  --train_domains arxiv wikipedia reddit wikihow \
  --train_generators chatGPT davinci cohere dolly \
  --val_generator flant5 \
  --epochs 3 \
  --batch_size 64 \
  --bf16 \
  --use_wandb \
  --run_name m4-english \
  --output_dir runs/m4-english
```

### Quick Test (3 min):
```bash
python train.py \
  --train_domains arxiv \
  --train_generators chatGPT \
  --val_generator davinci \
  --max_train_samples 200 \
  --max_val_samples 50 \
  --epochs 1 \
  --batch_size 8 \
  --output_dir runs/test
```

### Larger Model (+1-2% F1, 8-10 hours):
```bash
python train.py \
  --model roberta-large \
  --lr 1e-5 \
  --train_domains arxiv wikipedia reddit wikihow \
  --train_generators chatGPT davinci cohere dolly \
  --val_generator flant5 \
  --epochs 3 \
  --batch_size 32 \
  --bf16 \
  --use_wandb \
  --run_name m4-large \
  --output_dir runs/m4-large
```

---

## Inference

```bash
python predict.py \
  --model_path runs/m4-english \
  --input_file test.jsonl \
  --output_file predictions.jsonl
```

---

## Important Notes

**English Domains Only:**
- arxiv (scientific papers)
- wikipedia (encyclopedia)
- reddit (social media)
- wikihow (how-to guides)
- peerread (academic reviews)

**Available Generators:**
- chatGPT, davinci, cohere, dolly, flant5, llama, bloomz

**Don't Use (broken):**
- arxiv_bloomz
- peerread_bloomz

---

**See [README.md](./README.md) for full documentation.**
