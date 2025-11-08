# Testing Your Trained AI Detector Model

**Model Performance:** 97.12% F1 Score (trained on M4 dataset)

---

## Quick Start

### 1. Pull the Model from GitHub

```bash
cd m4-training
git pull
```

Your model is now in `final-model/` directory.

### 2. Install Dependencies

```bash
pip install transformers torch
```

---

## Test the Model

### Method 1: Quick Python Test

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load your trained model
model = AutoModelForSequenceClassification.from_pretrained("./final-model")
tokenizer = AutoTokenizer.from_pretrained("./final-model")

# Test text (try both human and AI-generated text)
test_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Quantum mechanics is a fundamental theory in physics that describes the behavior of matter and energy at atomic and subatomic scales.",
    "In this comprehensive analysis, we will delve into the multifaceted aspects of artificial intelligence and its transformative impact on modern society."
]

# Detect AI for each text
for text in test_texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=1)

    human_prob = predictions[0][0].item()
    ai_prob = predictions[0][1].item()

    label = "HUMAN" if human_prob > ai_prob else "AI-GENERATED"
    confidence = max(human_prob, ai_prob)

    print(f"\nText: {text[:60]}...")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Human: {human_prob:.2%} | AI: {ai_prob:.2%}")
```

### Method 2: Using the Prediction Script

If you have `predict.py` in the repo:

```bash
# Create test file
echo '{"text": "This is a test sentence to analyze."}' > test.jsonl

# Run prediction
python predict.py \
  --model_path final-model \
  --input_file test.jsonl \
  --output_file predictions.jsonl

# View results
cat predictions.jsonl
```

---

## Understanding the Output

### Labels:
- **Label 0 / HUMAN**: Text is likely written by a human
- **Label 1 / AI-GENERATED**: Text is likely AI-generated

### Confidence Scores:
- **>90%**: Very confident prediction
- **70-90%**: Confident prediction
- **50-70%**: Uncertain (close call)

---

## Testing Tips

### Good Test Cases:

**Human-written text:**
- News articles (pre-2020)
- Academic papers from ArXiv (pre-2020)
- Reddit posts (casual writing)
- Personal emails/messages

**AI-generated text:**
- ChatGPT responses
- Claude responses
- Copy from AI writing tools
- Generate with: https://chat.openai.com

### What Makes Detection Hard:
- Very short texts (<50 words)
- Heavily edited AI text
- Technical jargon (both human and AI use it)
- Lists and structured data

---

## Example: Create a Simple CLI Tool

Save as `detect.py`:

```python
#!/usr/bin/env python3
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model
print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained("./final-model")
tokenizer = AutoTokenizer.from_pretrained("./final-model")

# Get text from command line or stdin
if len(sys.argv) > 1:
    text = " ".join(sys.argv[1:])
else:
    print("Enter text to analyze (Ctrl+D when done):")
    text = sys.stdin.read()

# Predict
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=1)

human_prob = predictions[0][0].item()
ai_prob = predictions[0][1].item()

# Display result
print("\n" + "="*50)
print(f"TEXT: {text[:200]}...")
print("="*50)
if ai_prob > human_prob:
    print(f"🤖 AI-GENERATED ({ai_prob:.1%} confidence)")
else:
    print(f"👤 HUMAN-WRITTEN ({human_prob:.1%} confidence)")
print(f"\nScores: Human={human_prob:.1%}, AI={ai_prob:.1%}")
```

**Use it:**
```bash
chmod +x detect.py

# Test with argument
python detect.py "This is my test sentence"

# Test with pipe
echo "Some text to analyze" | python detect.py
```

---

## Model Details

- **Base Model:** RoBERTa-base
- **Training Data:** M4 dataset (English only)
- **Domains:** arxiv, wikipedia, reddit, wikihow
- **Generators Seen:** chatGPT, davinci, cohere, dolly
- **Validation Generator:** flant5 (unseen, open-set)
- **Performance:** 97.12% F1, 97.17% accuracy
- **Max Input Length:** 512 tokens (~400 words)
- **Longer texts:** Automatically chunked by predict.py

---

## Next Steps

1. **Test with your own text** - Try various sources
2. **Integrate into your app** - Use the model in production
3. **Fine-tune further** - Add your own domain-specific data
4. **Deploy** - Host as API with FastAPI/Flask

---

**Questions? Check the main README.md or ask Claude Code!** 🚀
