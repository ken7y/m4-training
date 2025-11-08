import pandas as pd
import json
from datetime import datetime
import re

print("="*60)
print("PREPARING REDDIT DATA FOR TRAINING")
print("="*60)

# Load WSB data
print("\n1. Loading WSB Reddit data...")
df = pd.read_csv("wsb/reddit_wsb.csv")
print(f"Total rows: {len(df)}")

# Check date range
print(f"\n2. Checking date range...")
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# Filter for pre-ChatGPT era (before Nov 30, 2022)
chatgpt_launch = pd.to_datetime('2022-11-30')
df_pre_chatgpt = df[df['timestamp'] < chatgpt_launch].copy()
print(f"Posts before ChatGPT launch: {len(df_pre_chatgpt)}/{len(df)} ({len(df_pre_chatgpt)/len(df)*100:.1f}%)")

# Clean the data
print(f"\n3. Cleaning data...")

# Remove rows with missing body
df_pre_chatgpt['body'] = df_pre_chatgpt['body'].fillna('')
initial_count = len(df_pre_chatgpt)

# Filter: body must have at least 100 characters (meaningful posts)
df_cleaned = df_pre_chatgpt[df_pre_chatgpt['body'].str.len() >= 100].copy()
print(f"   Removed posts with body <100 chars: {initial_count - len(df_cleaned)}")

# Remove URLs from body text (but keep the text)
def clean_text(text):
    if pd.isna(text):
        return ""
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove excessive newlines
    text = re.sub(r'\n+', '\n', text)
    # Remove excessive spaces
    text = re.sub(r'\s+', ' ', text)
    # Strip
    text = text.strip()
    return text

df_cleaned['body_clean'] = df_cleaned['body'].apply(clean_text)

# Remove if cleaned text is now too short
df_cleaned = df_cleaned[df_cleaned['body_clean'].str.len() >= 100].copy()
print(f"   After cleaning: {len(df_cleaned)} posts remain")

# Remove duplicates
df_cleaned = df_cleaned.drop_duplicates(subset=['body_clean'])
print(f"   After removing duplicates: {len(df_cleaned)} posts")

# Sample distribution across time to avoid temporal bias
print(f"\n4. Sampling posts across time periods...")
df_cleaned['year_month'] = df_cleaned['timestamp'].dt.to_period('M')
posts_per_month = df_cleaned.groupby('year_month').size()
print(f"   Date range: {df_cleaned['timestamp'].min()} to {df_cleaned['timestamp'].max()}")
print(f"   Unique months: {len(posts_per_month)}")

# Sample evenly across months (max 200 per month to avoid over-representation)
sampled_dfs = []
for period, group in df_cleaned.groupby('year_month'):
    sample_size = min(200, len(group))
    sampled = group.sample(n=sample_size, random_state=42)
    sampled_dfs.append(sampled)

df_sampled = pd.concat(sampled_dfs, ignore_index=True)
print(f"   Total sampled posts: {len(df_sampled)}")

# Create training format (JSONL format like M4)
print(f"\n5. Creating training format...")

training_data = []
for idx, row in df_sampled.iterrows():
    # Reddit posts are human-written
    training_data.append({
        'text': row['body_clean'],
        'label': 0,  # 0 = human
        'source': 'reddit_wsb',
        'timestamp': str(row['timestamp']),
        'id': row['id']
    })

# Save as JSONL
output_file = 'reddit_wsb_training.jsonl'
with open(output_file, 'w') as f:
    for item in training_data:
        f.write(json.dumps(item) + '\n')

print(f"\n6. Saved to {output_file}")

# Statistics
print(f"\n{'='*60}")
print(f"FINAL STATISTICS")
print(f"{'='*60}")
print(f"Total human posts (Reddit WSB): {len(training_data)}")
print(f"Date range: {df_sampled['timestamp'].min()} to {df_sampled['timestamp'].max()}")
print(f"Average post length: {df_sampled['body_clean'].str.len().mean():.0f} characters")
print(f"Min post length: {df_sampled['body_clean'].str.len().min()}")
print(f"Max post length: {df_sampled['body_clean'].str.len().max()}")

# Show sample posts
print(f"\n{'='*60}")
print(f"SAMPLE POSTS")
print(f"{'='*60}")
for i in range(min(3, len(df_sampled))):
    sample = df_sampled.iloc[i]
    print(f"\n[{i+1}] Date: {sample['timestamp']}")
    print(f"Text: {sample['body_clean'][:300]}...")
    print(f"Length: {len(sample['body_clean'])} chars")

print(f"\n{'='*60}")
print(f"✅ Reddit training data prepared!")
print(f"{'='*60}")

# Save summary CSV for inspection
df_sampled[['timestamp', 'body_clean', 'id']].to_csv('reddit_wsb_training_summary.csv', index=False)
print(f"\nSummary saved to: reddit_wsb_training_summary.csv")
print(f"Training data saved to: {output_file}")
