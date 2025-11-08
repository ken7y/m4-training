import pandas as pd

# Load AI-generated data
print("Loading AI_Generated.csv...")
ai_df = pd.read_csv("dataset_kaggle_ai/AI_Generated.csv")
print(f"AI data rows: {len(ai_df)}")
print(f"AI columns: {ai_df.columns.tolist()}")

# Load human-written data
print("\nLoading Human-Written.csv...")
human_df = pd.read_csv("dataset_kaggle_ai/Human-Written.csv")
print(f"Human data rows: {len(human_df)}")
print(f"Human columns: {human_df.columns.tolist()}")

# Extract text from AI data (use 'Generated' column)
ai_texts = []
for idx, row in ai_df.iterrows():
    text = row['Generated']
    if pd.notna(text) and len(str(text).strip()) > 50:  # Filter out empty/short texts
        ai_texts.append({
            'text': str(text).strip(),
            'label': 1  # 1 = AI
        })

print(f"\nExtracted {len(ai_texts)} valid AI texts")

# Extract text from human data (use 'Text' column)
human_texts = []
for idx, row in human_df.iterrows():
    text = row['Text']
    if pd.notna(text) and len(str(text).strip()) > 50:  # Filter out empty/short texts
        human_texts.append({
            'text': str(text).strip(),
            'label': 0  # 0 = Human
        })

print(f"Extracted {len(human_texts)} valid human texts")

# Combine and sample 500 total (250 AI + 250 Human if possible)
import random
random.seed(42)

# Sample from each
num_samples = 250
ai_sample = random.sample(ai_texts, min(num_samples, len(ai_texts)))
human_sample = random.sample(human_texts, min(num_samples, len(human_texts)))

# If we don't have enough human samples, fill with more AI samples
total_needed = 500
current_total = len(ai_sample) + len(human_sample)
if current_total < total_needed:
    remaining_ai = [x for x in ai_texts if x not in ai_sample]
    additional_needed = total_needed - current_total
    ai_sample.extend(random.sample(remaining_ai, min(additional_needed, len(remaining_ai))))

# Combine
combined = ai_sample + human_sample
random.shuffle(combined)

# Create dataframe
df = pd.DataFrame(combined)

print(f"\n{'='*60}")
print(f"Final dataset:")
print(f"Total samples: {len(df)}")
print(f"AI samples: {(df['label'] == 1).sum()}")
print(f"Human samples: {(df['label'] == 0).sum()}")
print(f"{'='*60}")

# Save
output_file = "kaggle_test_dataset.csv"
df.to_csv(output_file, index=False)
print(f"\nSaved to {output_file}")

# Show first few samples
print(f"\nFirst 3 samples:")
for i in range(min(3, len(df))):
    print(f"\n[{i+1}] Label: {'AI' if df.iloc[i]['label'] == 1 else 'Human'}")
    print(f"Text: {df.iloc[i]['text'][:200]}...")
