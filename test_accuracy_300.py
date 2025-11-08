import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model
print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained("./final-model")
tokenizer = AutoTokenizer.from_pretrained("./final-model")
model.eval()

# Load CSV
print("Loading CSV...")
df = pd.read_csv("balanced_ai_human_prompts.csv")
print(f"Total entries: {len(df)}")

# Test on 300 samples
sample_size = 300
df_sample = df.sample(n=sample_size, random_state=42)

print(f"\nTesting on {sample_size} samples...")

predictions = []
actuals = []

for idx, row in df_sample.iterrows():
    text = row['text']
    actual_label = row['generated']  # 1 = AI, 0 = Human

    if (idx + 1) % 100 == 0:
        print(f"Processing {idx + 1}/{sample_size}...")

    # Predict
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    # Index 0 = Human, Index 1 = AI
    ai_prob = probs[0][1].item()

    # Round to extreme: >0.5 = AI (1), <=0.5 = Human (0)
    predicted_label = 1 if ai_prob > 0.5 else 0

    predictions.append(predicted_label)
    actuals.append(actual_label)

# Calculate accuracy
correct = sum([1 for p, a in zip(predictions, actuals) if p == a])
accuracy = (correct / len(predictions)) * 100

print(f"\n{'='*60}")
print(f"ACCURACY RESULTS (Sample of {sample_size})")
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
print(f"{'='*60}")
