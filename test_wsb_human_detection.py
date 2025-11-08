import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import random

# Load WSB data
print("Loading WSB Reddit data...")
df = pd.read_csv("wsb/reddit_wsb.csv")
print(f"Total rows: {len(df)}")

# Filter for posts with body text (not empty, not title-only posts)
df['body'] = df['body'].fillna('')
df_with_body = df[df['body'].str.len() > 50].copy()  # At least 50 chars
print(f"Posts with body text (>50 chars): {len(df_with_body)}")

# Sample 1000 posts
random.seed(42)
sample_size = min(1000, len(df_with_body))
sampled_posts = df_with_body.sample(n=sample_size, random_state=42)

print(f"\nSampled {sample_size} posts for testing")
print(f"All posts are HUMAN (label=0)")

# Save sample for reference
sampled_posts[['title', 'body']].to_csv('wsb_sample_1000.csv', index=False)
print(f"Sample saved to wsb_sample_1000.csv")

results = {}

# ============================================================
# TEST MODEL 1: M4 (Your trained model)
# ============================================================
print(f"\n{'='*60}")
print("TESTING M4 MODEL")
print(f"{'='*60}")

model = AutoModelForSequenceClassification.from_pretrained("./final-model")
tokenizer = AutoTokenizer.from_pretrained("./final-model")
model.eval()

correct_human = 0
total = 0

for idx, row in sampled_posts.iterrows():
    text = row['body']

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    ai_prob = probs[0][1].item()
    predicted_label = 1 if ai_prob > 0.5 else 0

    if predicted_label == 0:  # Correctly identified as human
        correct_human += 1
    total += 1

    if (total) % 100 == 0:
        print(f"Processed {total}/{sample_size}...")

human_accuracy = (correct_human / total) * 100

results['M4'] = {
    'correct_human': correct_human,
    'total': total,
    'human_accuracy': human_accuracy
}

print(f"\nM4 Results:")
print(f"Correctly identified as HUMAN: {correct_human}/{total} ({human_accuracy:.2f}%)")
print(f"Incorrectly flagged as AI: {total - correct_human}/{total} ({100 - human_accuracy:.2f}%)")

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

correct_human = 0
total = 0

for idx, row in sampled_posts.iterrows():
    text = row['body']

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    predicted_class = torch.argmax(probs, dim=-1).item()

    if predicted_class == 0:  # Correctly identified as human
        correct_human += 1
    total += 1

    if (total) % 100 == 0:
        print(f"Processed {total}/{sample_size}...")

human_accuracy = (correct_human / total) * 100

results['HC3'] = {
    'correct_human': correct_human,
    'total': total,
    'human_accuracy': human_accuracy
}

print(f"\nHC3 Results:")
print(f"Correctly identified as HUMAN: {correct_human}/{total} ({human_accuracy:.2f}%)")
print(f"Incorrectly flagged as AI: {total - correct_human}/{total} ({100 - human_accuracy:.2f}%)")

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

correct_human = 0
total = 0

for idx, row in sampled_posts.iterrows():
    text = row['body']

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    predicted_class = torch.argmax(probs, dim=-1).item()

    if predicted_class == 0:  # Correctly identified as human
        correct_human += 1
    total += 1

    if (total) % 100 == 0:
        print(f"Processed {total}/{sample_size}...")

human_accuracy = (correct_human / total) * 100

results['SimpleAI'] = {
    'correct_human': correct_human,
    'total': total,
    'human_accuracy': human_accuracy
}

print(f"\nSimpleAI Results:")
print(f"Correctly identified as HUMAN: {correct_human}/{total} ({human_accuracy:.2f}%)")
print(f"Incorrectly flagged as AI: {total - correct_human}/{total} ({100 - human_accuracy:.2f}%)")

del model, tokenizer

# ============================================================
# FINAL COMPARISON
# ============================================================
print(f"\n{'='*60}")
print("FINAL COMPARISON - WSB HUMAN TEXT DETECTION")
print(f"{'='*60}")
print(f"Dataset: {sample_size} Reddit WSB posts (ALL HUMAN)")
print(f"{'='*60}")
print(f"{'Model':<12} | {'Correct Human':>15} | {'Human Accuracy':>15}")
print(f"{'-'*60}")
for model_name, metrics in results.items():
    print(f"{model_name:<12} | {metrics['correct_human']:>7}/{metrics['total']:>6} | {metrics['human_accuracy']:>14.2f}%")
print(f"{'='*60}")

# Determine winner
best_model = max(results.items(), key=lambda x: x[1]['human_accuracy'])
print(f"\n🏆 BEST AT DETECTING HUMAN TEXT: {best_model[0]} ({best_model[1]['human_accuracy']:.2f}%)")

# Worst model (most false positives)
worst_model = min(results.items(), key=lambda x: x[1]['human_accuracy'])
print(f"❌ WORST (Most false positives): {worst_model[0]} ({100 - worst_model[1]['human_accuracy']:.2f}% false positive rate)")
