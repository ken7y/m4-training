import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
print("Loading SimpleAI ChatGPT Detector model...")
tokenizer = AutoTokenizer.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta")
model = AutoModelForSequenceClassification.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta")
model.eval()

# Load CSV
print("Loading CSV...")
df = pd.read_csv("balanced_ai_human_prompts.csv")
print(f"Total entries: {len(df)}")

# Test on full dataset
sample_size = len(df)
df_sample = df

print(f"\nTesting on {sample_size} samples (full dataset)...")

predictions = []
actuals = []

for idx, row in df_sample.iterrows():
    text = row['text']
    actual_label = row['generated']  # 1 = AI, 0 = Human

    if (idx + 1) % 100 == 0:
        print(f"Processing {idx + 1}/{sample_size}...")

    # Predict
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Get prediction
    # Check model config to see which index is which
    # Typically: 0=human, 1=AI or 0=fake, 1=real
    predicted_class = torch.argmax(probs, dim=-1).item()

    predictions.append(predicted_class)
    actuals.append(actual_label)

# Calculate accuracy
correct = sum([1 for p, a in zip(predictions, actuals) if p == a])
accuracy = (correct / len(predictions)) * 100

print(f"\n{'='*60}")
print(f"SIMPLEAI MODEL ACCURACY RESULTS (Sample of {sample_size})")
print(f"{'='*60}")
print(f"Correct predictions: {correct}/{len(predictions)}")
print(f"Accuracy: {accuracy:.2f}%")

# Breakdown
ai_correct = sum([1 for p, a in zip(predictions, actuals) if p == a and a == 1])
human_correct = sum([1 for p, a in zip(predictions, actuals) if p == a and a == 0])
ai_total = sum([1 for a in actuals if a == 1])
human_total = sum([1 for a in actuals if a == 0])

print(f"\nBreakdown:")
print(f"AI Detection: {ai_correct}/{ai_total} correct ({(ai_correct/ai_total*100) if ai_total > 0 else 0:.2f}%)")
print(f"Human Detection: {human_correct}/{human_total} correct ({(human_correct/human_total*100) if human_total > 0 else 0:.2f}%)")

# Confusion matrix info
ai_as_human = sum([1 for p, a in zip(predictions, actuals) if p == 0 and a == 1])  # False negatives
human_as_ai = sum([1 for p, a in zip(predictions, actuals) if p == 1 and a == 0])  # False positives

print(f"\nErrors:")
print(f"AI misclassified as Human: {ai_as_human}")
print(f"Human misclassified as AI: {human_as_ai}")
print(f"{'='*60}")

print(f"\n{'='*60}")
print(f"COMPARISON TO OTHER MODELS:")
print(f"{'='*60}")
print(f"M4 Model Accuracy: 75.67%")
print(f"M4 AI Detection: 100.00% | M4 Human Detection: 51.35%")
print(f"\nHC3 Model Accuracy: 39.35%")
print(f"HC3 AI Detection: 4.73% | HC3 Human Detection: 73.96%")
print(f"\nSimpleAI Model Accuracy: {accuracy:.2f}%")
print(f"SimpleAI AI Detection: {(ai_correct/ai_total*100) if ai_total > 0 else 0:.2f}% | SimpleAI Human Detection: {(human_correct/human_total*100) if human_total > 0 else 0:.2f}%")
print(f"{'='*60}")
