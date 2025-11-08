import pandas as pd
import json
import re
import html

print("="*60)
print("PROCESSING FULL REDDIT BIGSET (1M+ posts)")
print("="*60)

def deep_clean(text):
    """Clean text removing all HTML, markdown, URLs"""
    # Decode HTML entities
    for _ in range(5):
        new_text = html.unescape(text)
        if new_text == text:
            break
        text = new_text

    # Replace <lb> with newline
    text = text.replace('<lb>', '\n')

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove markdown links
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'\[[^\]]*\]', '', text)

    # Remove URLs aggressively
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'\S+\.(com|org|net|edu|gov)\S*', '', text)

    # Remove markdown formatting
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^\*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    text = re.sub(r'~~([^~]+)~~', r'\1', text)

    # Remove markdown headers
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)

    # Remove code blocks
    text = re.sub(r'```[^`]*```', '', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)

    # Remove HTML entities
    text = re.sub(r'&[a-zA-Z0-9#]+;', '', text)

    # Clean whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    text = text.strip()

    return text

# Process in chunks and write to JSONL
output_file = 'reddit_bigset_full_training.jsonl'
chunksize = 50000
processed_count = 0
saved_count = 0

print(f"\nProcessing and cleaning posts...")
print(f"Writing to: {output_file}")

with open(output_file, 'w') as out_f:
    for chunk_num, chunk in enumerate(pd.read_csv('reddit-bigset/rspct.tsv', sep='\t', chunksize=chunksize), 1):
        # Clean
        chunk['selftext_clean'] = chunk['selftext'].apply(deep_clean)

        # Filter: keep if >= 100 chars after cleaning
        chunk = chunk[chunk['selftext_clean'].str.len() >= 100]

        # Write to JSONL
        for idx, row in chunk.iterrows():
            item = {
                'text': row['selftext_clean'],
                'label': 0,  # human
                'source': f"reddit_{row['subreddit']}",
                'id': row['id']
            }
            out_f.write(json.dumps(item) + '\n')
            saved_count += 1

        processed_count += len(chunk)

        if chunk_num % 5 == 0:
            print(f"Processed {chunk_num * chunksize:,} rows, saved {saved_count:,} posts...")

print(f"\n{'='*60}")
print(f"PROCESSING COMPLETE")
print(f"{'='*60}")
print(f"Total posts saved: {saved_count:,}")
print(f"File: {output_file}")
print(f"{'='*60}")
