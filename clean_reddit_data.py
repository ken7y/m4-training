import json
import re
import html

print("="*60)
print("CLEANING REDDIT DATA")
print("="*60)

# Load the training data
with open('reddit_wsb_training.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

print(f"\nTotal posts: {len(data)}")

def deep_clean_text(text):
    """Remove all HTML, markdown, and special formatting"""

    # Decode HTML entities (&nbsp; &gt; &#123; etc)
    text = html.unescape(text)

    # Remove HTML tags (anything in < >)
    text = re.sub(r'<[^>]+>', '', text)

    # Remove markdown links [text](url) -> just keep the text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # Remove standalone URLs (http, https, www)
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove markdown formatting
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)  # **bold** -> bold
    text = re.sub(r'\*([^\*]+)\*', r'\1', text)      # *italic* -> italic
    text = re.sub(r'__([^_]+)__', r'\1', text)       # __bold__ -> bold
    text = re.sub(r'_([^_]+)_', r'\1', text)         # _italic_ -> italic
    text = re.sub(r'~~([^~]+)~~', r'\1', text)       # ~~strikethrough~~ -> strikethrough

    # Remove markdown headers (# ## ###)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)

    # Remove code blocks (```)
    text = re.sub(r'```[^`]*```', '', text)

    # Remove inline code (`code`)
    text = re.sub(r'`([^`]+)`', r'\1', text)

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove excessive newlines
    text = re.sub(r'\n+', '\n', text)

    # Strip
    text = text.strip()

    return text

# Clean all posts
print("\nCleaning posts...")
cleaned_count = 0
removed_count = 0

cleaned_data = []
for item in data:
    original_text = item['text']
    cleaned_text = deep_clean_text(original_text)

    # Keep only if still has meaningful content (>100 chars)
    if len(cleaned_text) >= 100:
        item['text'] = cleaned_text
        cleaned_data.append(item)
        cleaned_count += 1
    else:
        removed_count += 1

print(f"Posts cleaned: {cleaned_count}")
print(f"Posts removed (too short after cleaning): {removed_count}")

# Save cleaned data
output_file = 'reddit_wsb_training_cleaned.jsonl'
with open(output_file, 'w') as f:
    for item in cleaned_data:
        f.write(json.dumps(item) + '\n')

print(f"\n{'='*60}")
print(f"CLEANING COMPLETE")
print(f"{'='*60}")
print(f"Original posts: {len(data)}")
print(f"Cleaned posts: {len(cleaned_data)}")
print(f"Removed: {removed_count}")
print(f"\nSaved to: {output_file}")

# Show before/after samples
print(f"\n{'='*60}")
print(f"BEFORE/AFTER SAMPLES")
print(f"{'='*60}")

# Reload original
with open('reddit_wsb_training.jsonl', 'r') as f:
    original_data = [json.loads(line) for line in f]

for i in range(min(3, len(cleaned_data))):
    orig = original_data[i]['text'][:300]
    clean = cleaned_data[i]['text'][:300]

    print(f"\n[{i+1}] BEFORE:")
    print(orig)
    print(f"\n[{i+1}] AFTER:")
    print(clean)
    print("-"*60)

# Check for remaining artifacts
print(f"\n{'='*60}")
print(f"VERIFICATION - CHECKING FOR REMAINING ARTIFACTS")
print(f"{'='*60}")

artifacts = {
    'HTML tags': 0,
    'HTML entities': 0,
    'Markdown links': 0,
    'URLs': 0,
}

for item in cleaned_data:
    text = item['text']

    if re.search(r'<[^>]+>', text):
        artifacts['HTML tags'] += 1
    if re.search(r'&[a-z]+;|&#\d+;', text, re.I):
        artifacts['HTML entities'] += 1
    if re.search(r'\[.*?\]\(.*?\)', text):
        artifacts['Markdown links'] += 1
    if re.search(r'http|www\.', text, re.I):
        artifacts['URLs'] += 1

for artifact, count in artifacts.items():
    status = "✅" if count == 0 else f"❌ {count} found"
    print(f"{artifact:<20}: {status}")

print(f"\n{'='*60}")
print(f"✅ Data cleaned and ready for training!")
print(f"{'='*60}")
