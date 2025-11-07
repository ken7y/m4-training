#!/usr/bin/env python3
"""
Quick test to verify chunking works on M4 data
"""

import json
import os
from transformers import AutoTokenizer

def normalize_digits(text):
    import re
    return re.sub(r'\d', '1', text)

def chunk_text_with_overlap(text, tokenizer, max_length=512, overlap=50):
    """Chunk text into overlapping segments"""
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    max_tokens_per_chunk = max_length - 2  # Account for [CLS] and [SEP]

    if len(tokens) <= max_tokens_per_chunk:
        return [text]

    chunks = []
    start_idx = 0

    while start_idx < len(tokens):
        end_idx = min(start_idx + max_tokens_per_chunk, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)

        if end_idx >= len(tokens):
            break
        start_idx += max_tokens_per_chunk - overlap

    return chunks

def test_m4_chunking(data_file, num_samples=10):
    """Test chunking on real M4 data"""
    print(f"Testing chunking on: {data_file}\n")
    print("="*80)

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    with open(data_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break

            item = json.loads(line.strip())
            # M4 has both human_text and machine_text, test with machine_text
            text = item.get('machine_text', item.get('text', item.get('content', '')))

            if not text:
                continue

            # Apply preprocessing
            text_cleaned = normalize_digits(text)

            # Get token count
            tokens = tokenizer.encode(text_cleaned, add_special_tokens=True)

            # Chunk if needed
            chunks = chunk_text_with_overlap(text_cleaned, tokenizer, max_length=512, overlap=50)

            # Print stats
            print(f"\n📄 Sample {i+1}:")
            print(f"  Original length: {len(text)} chars")
            print(f"  Token count: {len(tokens)} tokens")
            print(f"  Needs chunking: {'YES ✂️' if len(tokens) > 512 else 'NO ✅'}")
            print(f"  Number of chunks: {len(chunks)}")

            if len(chunks) > 1:
                print(f"  Chunk sizes: {[len(tokenizer.encode(c, add_special_tokens=True)) for c in chunks]} tokens")
                print(f"  Preview chunk 1 (first 100 chars): {chunks[0][:100]}...")
                print(f"  Preview chunk 2 (first 100 chars): {chunks[1][:100]}...")
            else:
                print(f"  Preview (first 200 chars): {text_cleaned[:200]}...")

            print("-" * 80)

    print("\n✅ Chunking test complete!")
    print("\n💡 Key points:")
    print("  - Texts >512 tokens automatically split into chunks")
    print("  - 50 token overlap between chunks (prevents context loss)")
    print("  - Each chunk becomes separate training example")
    print("  - During inference, predictions averaged across chunks")

if __name__ == "__main__":
    # Test on arxiv data (typically has longer texts)
    data_file = "M4/data/arxiv_chatGPT.jsonl"

    if os.path.exists(data_file):
        test_m4_chunking(data_file, num_samples=10)
    else:
        print(f"❌ File not found: {data_file}")
        print("Make sure you're in the m4-training directory with M4 data cloned")
