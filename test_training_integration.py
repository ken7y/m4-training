#!/usr/bin/env python3
"""
Full integration test: Verify the complete training pipeline works
"""

import os
import sys
import json

# Add current dir to path
sys.path.insert(0, os.path.dirname(__file__))

# Import from train.py
from train import load_m4_data, normalize_digits, chunk_text_with_overlap
from transformers import AutoTokenizer

print("="*80)
print("FULL INTEGRATION TEST: M4 Training Pipeline")
print("="*80)

# 1. Test data loading
print("\n1️⃣ Testing data loading...")
print("-"*80)

data_dir = "M4/data"
domains = ["arxiv"]
generators = ["chatGPT"]

dataset = load_m4_data(
    data_dir=data_dir,
    domains=domains,
    generators=generators,
    normalize_digits_flag=True,
    max_samples=20  # Small sample
)

print(f"\n✅ Loaded {len(dataset)} examples")

# 2. Verify data structure
print("\n2️⃣ Verifying data structure...")
print("-"*80)

sample = dataset[0]
print(f"Sample keys: {list(sample.keys())}")
print(f"  text: {sample['text'][:100]}...")
print(f"  label: {sample['label']} ({'human' if sample['label'] == 0 else 'machine'})")
print(f"  generator: {sample['generator']}")
print(f"  domain: {sample['domain']}")

# Check labels are correct
labels = [item['label'] for item in dataset]
human_count = sum(1 for l in labels if l == 0)
machine_count = sum(1 for l in labels if l == 1)

print(f"\n📊 Label distribution:")
print(f"  Human (label=0): {human_count}")
print(f"  Machine (label=1): {machine_count}")

if human_count == machine_count:
    print("  ✅ BALANCED (as expected from M4 format)")
else:
    print("  ⚠️  IMBALANCED (unexpected!)")

# 3. Check for field name leakage
print("\n3️⃣ Checking for field name leakage...")
print("-"*80)

leakage_found = False
for i, item in enumerate(dataset):
    if 'human_text:' in item['text'] or 'machine_text:' in item['text']:
        print(f"❌ LEAKAGE in example {i}: {item['text'][:200]}")
        leakage_found = True
        break

if not leakage_found:
    print("✅ No field name leakage found in text data")

# 4. Test chunking
print("\n4️⃣ Testing chunking...")
print("-"*80)

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# Find a long text
long_text = max(dataset, key=lambda x: len(x['text']))['text']
print(f"Longest text: {len(long_text)} chars")

tokens = tokenizer.encode(long_text, add_special_tokens=True)
print(f"Tokens: {len(tokens)}")

chunks = chunk_text_with_overlap(long_text, tokenizer, max_length=512, overlap=50)
print(f"Chunks created: {len(chunks)}")
print(f"Chunk token sizes: {[len(tokenizer.encode(c, add_special_tokens=True)) for c in chunks]}")

if len(chunks) > 0 and all(len(tokenizer.encode(c, add_special_tokens=True)) <= 512 for c in chunks):
    print("✅ All chunks within 512 token limit")
else:
    print("❌ Some chunks exceed 512 tokens!")

# 5. Verify digit normalization
print("\n5️⃣ Verifying digit normalization...")
print("-"*80)

# Check if any digits remain
has_digits = any(any(c.isdigit() for c in item['text']) for item in dataset)

if not has_digits:
    print("✅ All digits normalized (replaced with '1')")
else:
    print("⚠️  Some digits found (this might be ok if normalization is disabled)")

# 6. Test label correctness
print("\n6️⃣ Testing label correctness...")
print("-"*80)

human_examples = [item for item in dataset if item['label'] == 0]
machine_examples = [item for item in dataset if item['label'] == 1]

print(f"Human examples (label=0):")
print(f"  Count: {len(human_examples)}")
print(f"  Generator: {human_examples[0]['generator'] if human_examples else 'N/A'}")

print(f"\nMachine examples (label=1):")
print(f"  Count: {len(machine_examples)}")
print(f"  Generator: {machine_examples[0]['generator'] if machine_examples else 'N/A'}")

if all(item['generator'] == 'human' for item in human_examples):
    print("✅ All human examples have generator='human'")
else:
    print("❌ Some human examples have wrong generator!")

if all(item['generator'] == 'chatGPT' for item in machine_examples):
    print("✅ All machine examples have correct generator")
else:
    print("❌ Some machine examples have wrong generator!")

# Final summary
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

all_checks = [
    len(dataset) == 20,  # Should be 10 lines × 2 = 20 examples
    human_count == machine_count,  # Balanced
    not leakage_found,  # No field name leakage
    len(chunks) > 0,  # Chunking works
    not has_digits,  # Digits normalized
    all(item['generator'] == 'human' for item in human_examples),  # Labels correct
]

if all(all_checks):
    print("✅ ALL CHECKS PASSED!")
    print("\n🚀 Training pipeline is ready to use!")
    print("\nYou can now run:")
    print("  python train.py --max_train_samples 1000 --max_val_samples 200 --epochs 1")
else:
    print("❌ SOME CHECKS FAILED!")
    print("\n🔧 Fix the issues above before training.")

print("="*80)
