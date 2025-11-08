import pandas as pd
import json
import re
import html

print("="*60)
print("CLEANING REDDIT BIGSET SAMPLE")
print("="*60)

# Load sample
df = pd.read_csv('reddit_bigset_sample.csv')
print(f"\nTotal posts: {len(df)}")

def deep_clean(text):
    # Decode HTML entities (multiple passes for nested)
    for _ in range(5):
        new_text = html.unescape(text)
        if new_text == text:
            break
        text = new_text

    # Replace <lb> with actual newline (Reddit format)
    text = text.replace('<lb>', '\n')

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove markdown links [text](url) -> just text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'\[[^\]]*\]', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Remove markdown formatting
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)  # bold
    text = re.sub(r'\*([^\*]+)\*', r'\1', text)      # italic
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    text = re.sub(r'~~([^~]+)~~', r'\1', text)       # strikethrough

    # Remove markdown headers
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)

    # Remove code blocks
    text = re.sub(r'```[^`]*```', '', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)

    # Remove remaining HTML entities
    text = re.sub(r'&[a-zA-Z0-9#]+;', '', text)

    # Clean whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    text = text.strip()

    return text

# Clean all posts
print("\nCleaning posts...")
df['selftext_clean'] = df['selftext'].apply(deep_clean)

# Filter: keep only if still >= 100 chars after cleaning
df_clean = df[df['selftext_clean'].str.len() >= 100].copy()

print(f"Posts after cleaning: {len(df_clean)}")
print(f"Removed (too short): {len(df) - len(df_clean)}")

# Save as JSONL for training
training_data = []
for idx, row in df_clean.iterrows():
    training_data.append({
        'text': row['selftext_clean'],
        'label': 0,  # 0 = human
        'source': f"reddit_{row['subreddit']}",
        'id': row['id']
    })

output_file = 'reddit_bigset_training.jsonl'
with open(output_file, 'w') as f:
    for item in training_data:
        f.write(json.dumps(item) + '\n')

print(f"\n{'='*60}")
print(f"VERIFICATION")
print(f"{'='*60}")

# Check for artifacts
artifacts = {'HTML': 0, 'URLs': 0, 'Markdown': 0, 'Entities': 0}
for item in training_data:
    if re.search(r'<[^>]+>', item['text']):
        artifacts['HTML'] += 1
    if re.search(r'http|www\.', item['text'], re.I):
        artifacts['URLs'] += 1
    if re.search(r'\[[^\]]*\]', item['text']):
        artifacts['Markdown'] += 1
    if re.search(r'&[a-z]+;', item['text'], re.I):
        artifacts['Entities'] += 1

for name, count in artifacts.items():
    status = "✅" if count == 0 else f"❌ {count}"
    print(f"{name:<12}: {status}")

print(f"\n{'='*60}")
print(f"FINAL STATS")
print(f"{'='*60}")
print(f"Clean posts: {len(training_data):,}")
print(f"Avg length: {df_clean['selftext_clean'].str.len().mean():.0f} chars")
print(f"Min length: {df_clean['selftext_clean'].str.len().min()}")
print(f"Max length: {df_clean['selftext_clean'].str.len().max()}")
print(f"\nSaved to: {output_file}")

# Show samples
print(f"\n{'='*60}")
print(f"CLEANED SAMPLES")
print(f"{'='*60}")
for i in range(3):
    print(f"\n[{i+1}] {df_clean.iloc[i]['selftext_clean'][:250]}...")
    print(f"Length: {len(df_clean.iloc[i]['selftext_clean'])} chars")
