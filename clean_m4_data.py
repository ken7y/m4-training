#!/usr/bin/env python3
"""
Clean M4 Dataset - Remove HTML, Markdown, and Formatting

This script applies the same aggressive cleaning used for Reddit data to M4 dataset:
- HTML tags and entities (including nested: &amp;amp; → &amp; → &)
- Markdown formatting ([text](url), **bold**, etc.)
- URLs (http://, www., .com domains)
- Code blocks and inline code
- Excessive whitespace and newlines

The M4 dataset has two text fields per line:
- human_text (academic papers, Wikipedia, etc.) - label 0
- machine_text (ChatGPT, Davinci, etc.) - label 1

We clean BOTH fields to ensure consistency.
"""

import json
import re
import html
import os
from pathlib import Path

print("="*80)
print("CLEANING M4 DATASET")
print("="*80)

def deep_clean_text(text):
    """
    Remove all HTML, markdown, URLs, and special formatting.
    Same cleaning function used for Reddit data.
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Decode HTML entities (&nbsp; &gt; &#123; etc) - apply multiple times for nested entities
    for _ in range(3):  # Handle nested HTML entities like &amp;amp;
        text = html.unescape(text)

    # Remove HTML tags (anything in < >)
    text = re.sub(r'<[^>]+>', '', text)

    # Remove markdown links [text](url) -> just keep the text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # Remove standalone URLs (http, https, www)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+\.com\S*', '', text)  # Also catch domain.com URLs

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

    # Remove LaTeX-style math ($ $)
    text = re.sub(r'\$[^\$]+\$', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove excessive newlines
    text = re.sub(r'\n+', '\n', text)

    # Strip
    text = text.strip()

    return text


def clean_m4_file(input_path, output_dir, min_length=100):
    """
    Clean a single M4 JSONL file.
    
    Args:
        input_path: Path to input JSONL file
        output_dir: Directory to save cleaned file
        min_length: Minimum character length after cleaning (default: 100)
    
    Returns:
        dict with statistics
    """
    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, filename)
    
    print(f"\n{'='*80}")
    print(f"Processing: {filename}")
    print(f"{'='*80}")
    
    stats = {
        'total': 0,
        'cleaned': 0,
        'skipped_short': 0,
        'skipped_error': 0,
        'human_only_removed': 0,
        'machine_only_removed': 0,
        'both_removed': 0
    }
    
    cleaned_data = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            stats['total'] += 1
            
            try:
                item = json.loads(line.strip())
                
                # Extract human and machine text (handle different formats)
                human_text = None
                machine_text = None
                
                if 'human_text' in item:
                    human_text = item['human_text']
                elif 'text' in item:
                    human_text = item['text']
                
                if 'machine_text' in item:
                    machine_text = item['machine_text']
                
                # Handle list format (PeerRead files)
                if isinstance(human_text, list):
                    human_text = '\n\n'.join(str(x) for x in human_text if x)
                if isinstance(machine_text, list):
                    machine_text = '\n\n'.join(str(x) for x in machine_text if x)
                
                # Clean both texts
                if human_text:
                    cleaned_human = deep_clean_text(human_text)
                else:
                    cleaned_human = ""
                    
                if machine_text:
                    cleaned_machine = deep_clean_text(machine_text)
                else:
                    cleaned_machine = ""
                
                # Check if texts meet minimum length
                human_ok = len(cleaned_human) >= min_length
                machine_ok = len(cleaned_machine) >= min_length
                
                # Only keep if BOTH texts are long enough
                if human_ok and machine_ok:
                    # Update item with cleaned text
                    if 'human_text' in item:
                        item['human_text'] = cleaned_human
                    elif 'text' in item:
                        item['text'] = cleaned_human
                    
                    if 'machine_text' in item:
                        item['machine_text'] = cleaned_machine
                    
                    cleaned_data.append(item)
                    stats['cleaned'] += 1
                else:
                    stats['skipped_short'] += 1
                    if not human_ok and not machine_ok:
                        stats['both_removed'] += 1
                    elif not human_ok:
                        stats['human_only_removed'] += 1
                    else:
                        stats['machine_only_removed'] += 1
                        
            except Exception as e:
                stats['skipped_error'] += 1
                print(f"  ⚠️  Error on line {line_num}: {e}")
                continue
    
    # Save cleaned data
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in cleaned_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Print statistics
    print(f"\nResults:")
    print(f"  Total lines:          {stats['total']:,}")
    print(f"  ✅ Cleaned:            {stats['cleaned']:,}")
    print(f"  ❌ Skipped (too short): {stats['skipped_short']:,}")
    if stats['human_only_removed'] > 0:
        print(f"     - Human text too short:   {stats['human_only_removed']:,}")
    if stats['machine_only_removed'] > 0:
        print(f"     - Machine text too short: {stats['machine_only_removed']:,}")
    if stats['both_removed'] > 0:
        print(f"     - Both texts too short:   {stats['both_removed']:,}")
    if stats['skipped_error'] > 0:
        print(f"  ⚠️  Errors:            {stats['skipped_error']:,}")
    
    retention_rate = (stats['cleaned'] / stats['total'] * 100) if stats['total'] > 0 else 0
    print(f"  Retention rate:       {retention_rate:.1f}%")
    print(f"\n  Saved to: {output_path}")
    
    return stats


def main():
    """Clean all M4 dataset files"""
    
    # Configuration
    input_dir = "M4/data"
    output_dir = "M4_cleaned/data"
    min_length = 100  # Same as Reddit cleaning
    
    # English-only files (formal academic text)
    english_files = [
        # ArXiv (scientific papers)
        "arxiv_bloomz.jsonl",
        "arxiv_chatGPT.jsonl",
        "arxiv_cohere.jsonl",
        "arxiv_davinci.jsonl",
        "arxiv_dolly.jsonl",
        "arxiv_flant5.jsonl",
        
        # Wikipedia
        "wikipedia_bloomz.jsonl",
        "wikipedia_chatgpt.jsonl",
        "wikipedia_cohere.jsonl",
        "wikipedia_davinci.jsonl",
        "wikipedia_dolly.jsonl",
        
        # WikiHow
        "wikihow_bloomz.jsonl",
        "wikihow_chatGPT.jsonl",
        "wikihow_cohere.jsonl",
        "wikihow_davinci.jsonl",
        "wikihow_dolly2.jsonl",
        
        # PeerRead (academic reviews)
        "peerread_bloomz.jsonl",
        "peerread_chatgpt.jsonl",
        "peerread_cohere.jsonl",
        "peerread_davinci.jsonl",
        "peerread_dolly.jsonl",
        "peerread_llama.jsonl",
        
        # Reddit (casual text - M4 version)
        "reddit_bloomz.jsonl",
        "reddit_chatGPT.jsonl",
        "reddit_cohere.jsonl",
        "reddit_davinci.jsonl",
        "reddit_dolly.jsonl",
        "reddit_flant5.jsonl",
    ]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📁 Input:  {input_dir}")
    print(f"📁 Output: {output_dir}")
    print(f"📏 Min length: {min_length} characters\n")
    
    # Process all files
    total_stats = {
        'files_processed': 0,
        'files_skipped': 0,
        'total_lines': 0,
        'total_cleaned': 0,
        'total_skipped': 0
    }
    
    for filename in english_files:
        input_path = os.path.join(input_dir, filename)
        
        if not os.path.exists(input_path):
            print(f"⚠️  File not found: {filename}")
            total_stats['files_skipped'] += 1
            continue
        
        stats = clean_m4_file(input_path, output_dir, min_length)
        
        total_stats['files_processed'] += 1
        total_stats['total_lines'] += stats['total']
        total_stats['total_cleaned'] += stats['cleaned']
        total_stats['total_skipped'] += stats['skipped_short'] + stats['skipped_error']
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"CLEANING COMPLETE - SUMMARY")
    print(f"{'='*80}")
    print(f"Files processed:      {total_stats['files_processed']}")
    print(f"Files skipped:        {total_stats['files_skipped']}")
    print(f"Total lines:          {total_stats['total_lines']:,}")
    print(f"✅ Total cleaned:      {total_stats['total_cleaned']:,}")
    print(f"❌ Total skipped:      {total_stats['total_skipped']:,}")
    
    if total_stats['total_lines'] > 0:
        retention_rate = (total_stats['total_cleaned'] / total_stats['total_lines'] * 100)
        print(f"Overall retention:    {retention_rate:.1f}%")
    
    print(f"\n📁 Cleaned files saved to: {output_dir}/")
    print(f"\n💡 Next steps:")
    print(f"   1. Update train.py to use 'M4_cleaned/data' as data_dir")
    print(f"   2. Run training on RunPod with cleaned data")
    print(f"   3. Compare model performance with/without cleaning")
    

if __name__ == "__main__":
    main()
