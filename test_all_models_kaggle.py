import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load test data
print("Loading test dataset...")
df = pd.read_csv("kaggle_test_dataset.csv")
print(f"Total samples: {len(df)}")
print(f"AI samples: {(df['label'] == 1).sum()}")
print(f"Human samples: {(df['label'] == 0).sum()}")

results = {}

# ============================================================
# TEST MODEL 1: M4 (Your trained model)
# ============================================================
print(f"\n{'='*60}")
print("TESTING M4 MODEL (Your trained model)")
print(f"{'='*60}")

model = AutoModelForSequenceClassification.from_pretrained("./final-model")
tokenizer = AutoTokenizer.from_pretrained("./final-model")
model.eval()

predictions = []
actuals = []

for idx, row in df.iterrows():
    text = row['text']
    actual_label = row['label']

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    ai_prob = probs[0][1].item()
    predicted_label = 1 if ai_prob > 0.5 else 0

    predictions.append(predicted_label)
    actuals.append(actual_label)

correct = sum([1 for p, a in zip(predictions, actuals) if p == a])
accuracy = (correct / len(predictions)) * 100
ai_correct = sum([1 for p, a in zip(predictions, actuals) if p == a and a == 1])
human_correct = sum([1 for p, a in zip(predictions, actuals) if p == a and a == 0])
ai_total = sum([1 for a in actuals if a == 1])
human_total = sum([1 for a in actuals if a == 0])

results['M4'] = {
    'accuracy': accuracy,
    'ai_detection': (ai_correct/ai_total*100) if ai_total > 0 else 0,
    'human_detection': (human_correct/human_total*100) if human_total > 0 else 0
}

print(f"Accuracy: {accuracy:.2f}%")
print(f"AI Detection: {ai_correct}/{ai_total} ({results['M4']['ai_detection']:.2f}%)")
print(f"Human Detection: {human_correct}/{human_total} ({results['M4']['human_detection']:.2f}%)")

del model, tokenizer

# ============================================================
# TEST MODEL 2: HC3
# ============================================================
print(f"\n{'='*60}")
print("TESTING HC3 MODEL")
print(f"{'='*60}")

tokenizer = AutoTokenizer.from_pretrained("VSAsteroid/ai-text-detector-hc3")
model = AutoModelForSequenceClassification.from_pretrained("VSAsteroid/ai-text-detector-hc3")
model.eval()

predictions = []
actuals = []

for idx, row in df.iterrows():
    text = row['text']
    actual_label = row['label']

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    predicted_class = torch.argmax(probs, dim=-1).item()

    predictions.append(predicted_class)
    actuals.append(actual_label)

correct = sum([1 for p, a in zip(predictions, actuals) if p == a])
accuracy = (correct / len(predictions)) * 100
ai_correct = sum([1 for p, a in zip(predictions, actuals) if p == a and a == 1])
human_correct = sum([1 for p, a in zip(predictions, actuals) if p == a and a == 0])
ai_total = sum([1 for a in actuals if a == 1])
human_total = sum([1 for a in actuals if a == 0])

results['HC3'] = {
    'accuracy': accuracy,
    'ai_detection': (ai_correct/ai_total*100) if ai_total > 0 else 0,
    'human_detection': (human_correct/human_total*100) if human_total > 0 else 0
}

print(f"Accuracy: {accuracy:.2f}%")
print(f"AI Detection: {ai_correct}/{ai_total} ({results['HC3']['ai_detection']:.2f}%)")
print(f"Human Detection: {human_correct}/{human_total} ({results['HC3']['human_detection']:.2f}%)")

del model, tokenizer

# ============================================================
# TEST MODEL 3: SimpleAI
# ============================================================
print(f"\n{'='*60}")
print("TESTING SIMPLEAI MODEL")
print(f"{'='*60}")

tokenizer = AutoTokenizer.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta")
model = AutoModelForSequenceClassification.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta")
model.eval()

predictions = []
actuals = []

for idx, row in df.iterrows():
    text = row['text']
    actual_label = row['label']

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    predicted_class = torch.argmax(probs, dim=-1).item()

    predictions.append(predicted_class)
    actuals.append(actual_label)

correct = sum([1 for p, a in zip(predictions, actuals) if p == a])
accuracy = (correct / len(predictions)) * 100
ai_correct = sum([1 for p, a in zip(predictions, actuals) if p == a and a == 1])
human_correct = sum([1 for p, a in zip(predictions, actuals) if p == a and a == 0])
ai_total = sum([1 for a in actuals if a == 1])
human_total = sum([1 for a in actuals if a == 0])

results['SimpleAI'] = {
    'accuracy': accuracy,
    'ai_detection': (ai_correct/ai_total*100) if ai_total > 0 else 0,
    'human_detection': (human_correct/human_total*100) if human_total > 0 else 0
}

print(f"Accuracy: {accuracy:.2f}%")
print(f"AI Detection: {ai_correct}/{ai_total} ({results['SimpleAI']['ai_detection']:.2f}%)")
print(f"Human Detection: {human_correct}/{human_total} ({results['SimpleAI']['human_detection']:.2f}%)")

del model, tokenizer

# ============================================================
# FINAL COMPARISON
# ============================================================
print(f"\n{'='*60}")
print("FINAL COMPARISON - KAGGLE DATASET")
print(f"{'='*60}")
print(f"Dataset: {len(df)} samples ({ai_total} AI, {human_total} Human)")
print(f"{'='*60}")
print(f"{'Model':<12} | {'Accuracy':>10} | {'AI Detect':>10} | {'Human Detect':>12}")
print(f"{'-'*60}")
for model_name, metrics in results.items():
    print(f"{model_name:<12} | {metrics['accuracy']:>9.2f}% | {metrics['ai_detection']:>9.2f}% | {metrics['human_detection']:>11.2f}%")
print(f"{'='*60}")

# Determine winner
best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
print(f"\n🏆 WINNER: {best_model[0]} with {best_model[1]['accuracy']:.2f}% accuracy")
