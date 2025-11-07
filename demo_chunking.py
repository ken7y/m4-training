#!/usr/bin/env python3
"""
Demonstrate chunking with a long synthetic text
"""

from transformers import AutoTokenizer
import re

def normalize_digits(text):
    return re.sub(r'\d', '1', text)

def chunk_text_with_overlap(text, tokenizer, max_length=512, overlap=50):
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    max_tokens_per_chunk = max_length - 2

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

# Create a long text (simulate a long reddit post or arxiv abstract)
long_text = """
The development of artificial intelligence has revolutionized numerous fields including natural language processing,
computer vision, and machine learning. """ * 100  # Repeat to make it long

long_text += """
In recent years, large language models such as GPT-3, GPT-4, ChatGPT, and others have demonstrated remarkable
capabilities in generating human-like text. These models are trained on massive datasets containing billions of
tokens from diverse sources including books, websites, scientific papers, and social media. The training process
involves unsupervised learning where the model learns to predict the next token in a sequence given the previous
tokens. This approach has proven to be highly effective at capturing complex patterns in natural language.
""" * 50  # More repetition

print("🧪 CHUNKING DEMONSTRATION\n")
print("="*80)

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# Show original stats
print(f"\n📝 Original Text:")
print(f"  Length: {len(long_text):,} characters")

tokens = tokenizer.encode(long_text, add_special_tokens=True)
print(f"  Tokens: {len(tokens):,} tokens")
print(f"  Exceeds 512 limit: {'YES! ✂️ CHUNKING NEEDED' if len(tokens) > 512 else 'NO'}")

# Apply chunking
print(f"\n✂️  Applying chunking (max_length=512, overlap=50)...")
chunks = chunk_text_with_overlap(long_text, tokenizer, max_length=512, overlap=50)

print(f"\n📊 Chunking Results:")
print(f"  Number of chunks created: {len(chunks)}")
print(f"  Chunk token sizes: {[len(tokenizer.encode(c, add_special_tokens=True)) for c in chunks]}")

# Show each chunk details
for i, chunk in enumerate(chunks):
    chunk_tokens = tokenizer.encode(chunk, add_special_tokens=True)
    print(f"\n  Chunk {i+1}:")
    print(f"    Tokens: {len(chunk_tokens)}")
    print(f"    Characters: {len(chunk)}")
    print(f"    Preview: {chunk[:100]}...")

# Demonstrate overlap
if len(chunks) > 1:
    print(f"\n🔗 Demonstrating Overlap:")
    print(f"  Last 50 chars of chunk 1: ...{chunks[0][-50:]}")
    print(f"  First 50 chars of chunk 2: {chunks[1][:50]}...")
    print(f"  ✅ Notice the overlap to preserve context!")

print(f"\n💡 What happens during training:")
print(f"  - Original text → {len(chunks)} training examples")
print(f"  - Each chunk gets same label (e.g., 'machine-generated')")
print(f"  - Model learns from all chunks independently")

print(f"\n💡 What happens during inference:")
print(f"  - Text chunked same way → {len(chunks)} chunks")
print(f"  - Model predicts on each chunk")
print(f"  - Predictions averaged: mean([prob1, prob2, ..., prob{len(chunks)}])")
print(f"  - Final decision based on average probability")

print(f"\n✅ This ensures long texts don't get truncated and lose information!")
print("="*80)
