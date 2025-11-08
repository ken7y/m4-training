import pandas as pd
import re

print("="*60)
print("SAMPLING REDDIT BIGSET DATA")
print("="*60)

# Read TSV in chunks (it's 880MB)
print("\nReading TSV file in chunks...")
chunksize = 100000
total_rows = 0
valid_rows = 0
empty_rows = 0
short_rows = 0

# First pass: count what we have
for chunk in pd.read_csv('reddit-bigset/rspct.tsv', sep='\t', chunksize=chunksize):
    total_rows += len(chunk)

    # Count empty
    empty_count = chunk['selftext'].isna().sum() + (chunk['selftext'] == '').sum()
    empty_rows += empty_count

    # Count short (<100 chars)
    chunk['selftext_len'] = chunk['selftext'].fillna('').astype(str).str.len()
    short_count = ((chunk['selftext_len'] > 0) & (chunk['selftext_len'] < 100)).sum()
    short_rows += short_count

    # Count valid
    valid_count = (chunk['selftext_len'] >= 100).sum()
    valid_rows += valid_count

    if total_rows % 100000 == 0:
        print(f"Processed {total_rows:,} rows...")

print(f"\n{'='*60}")
print(f"INITIAL STATS")
print(f"{'='*60}")
print(f"Total rows: {total_rows:,}")
print(f"Empty selftext: {empty_rows:,} ({empty_rows/total_rows*100:.1f}%)")
print(f"Short selftext (<100 chars): {short_rows:,} ({short_rows/total_rows*100:.1f}%)")
print(f"Valid selftext (>=100 chars): {valid_rows:,} ({valid_rows/total_rows*100:.1f}%)")

# Sample 10,000 valid posts
print(f"\n{'='*60}")
print(f"SAMPLING VALID POSTS")
print(f"{'='*60}")
print(f"Sampling 10,000 posts with selftext >= 100 chars...")

sampled_posts = []
sample_target = 10000

for chunk in pd.read_csv('reddit-bigset/rspct.tsv', sep='\t', chunksize=chunksize):
    # Filter valid posts
    chunk['selftext'] = chunk['selftext'].fillna('')
    chunk = chunk[chunk['selftext'].str.len() >= 100]

    if len(chunk) > 0:
        sampled_posts.append(chunk)

    if sum(len(df) for df in sampled_posts) >= sample_target:
        break

# Combine and sample exactly 10k
df_combined = pd.concat(sampled_posts, ignore_index=True)
df_sampled = df_combined.sample(n=min(sample_target, len(df_combined)), random_state=42)

print(f"Sampled {len(df_sampled):,} posts")

# Show sample
print(f"\n{'='*60}")
print(f"SAMPLE POSTS")
print(f"{'='*60}")
for i in range(3):
    print(f"\n[{i+1}] Subreddit: {df_sampled.iloc[i]['subreddit']}")
    print(f"Title: {df_sampled.iloc[i]['title']}")
    print(f"Body: {df_sampled.iloc[i]['selftext'][:200]}...")
    print(f"Length: {len(df_sampled.iloc[i]['selftext'])} chars")

# Save sample
output_file = 'reddit_bigset_sample.csv'
df_sampled.to_csv(output_file, index=False)
print(f"\n{'='*60}")
print(f"✅ Saved {len(df_sampled):,} posts to {output_file}")
print(f"{'='*60}")
