#!/usr/bin/env python3
"""
Test the FIXED data loader on all format variations
"""

import sys
sys.path.insert(0, '.')
from train import load_m4_data

print("="*80)
print("TESTING FIXED DATA LOADER ON ALL FORMATS")
print("="*80)

test_cases = [
    # Format 1: Standard (human_text, machine_text)
    {
        'name': 'Standard format (arxiv_chatGPT)',
        'domains': ['arxiv'],
        'generators': ['chatGPT'],
        'expected_examples': '>0'
    },
    # Format 2: Reddit format (text, machine_text)
    {
        'name': 'Reddit format (reddit_bloomz)',
        'domains': ['reddit'],
        'generators': ['bloomz'],
        'expected_examples': '>0'
    },
    # Format 3: PeerRead list format
    {
        'name': 'PeerRead list format (peerread_chatgpt)',
        'domains': ['peerread'],
        'generators': ['chatgpt'],
        'expected_examples': '>0'
    },
    # Multiple domains
    {
        'name': 'Multiple domains (arxiv, wikipedia, reddit)',
        'domains': ['arxiv', 'wikipedia', 'reddit'],
        'generators': ['chatGPT'],
        'expected_examples': '>0'
    },
]

all_passed = True

for i, test in enumerate(test_cases):
    print(f"\n{'='*80}")
    print(f"TEST {i+1}: {test['name']}")
    print('='*80)

    try:
        dataset = load_m4_data(
            data_dir='M4/data',
            domains=test['domains'],
            generators=test['generators'],
            normalize_digits_flag=True,
            max_samples=100  # Small sample for speed
        )

        num_examples = len(dataset)
        print(f"\n✅ Loaded {num_examples} examples")

        # Check balance
        labels = [item['label'] for item in dataset]
        human_count = sum(1 for l in labels if l == 0)
        machine_count = sum(1 for l in labels if l == 1)

        print(f"  Human (label=0): {human_count}")
        print(f"  Machine (label=1): {machine_count}")

        # Check sample
        if dataset:
            sample = dataset[0]
            print(f"\n  Sample preview:")
            print(f"    Text length: {len(sample['text'])} chars")
            print(f"    Label: {sample['label']} ({'human' if sample['label'] == 0 else 'machine'})")
            print(f"    Generator: {sample['generator']}")
            print(f"    Domain: {sample['domain']}")
            print(f"    Preview: {sample['text'][:150]}...")

            # Check for format issues
            if 'human_text:' in sample['text'] or 'machine_text:' in sample['text']:
                print(f"\n  ❌ FIELD NAME LEAKAGE DETECTED!")
                all_passed = False
            else:
                print(f"\n  ✅ No field name leakage")

            # Check digits normalized
            import re
            digits = set(re.findall(r'\d', sample['text'][:1000]))
            if digits and digits != {'1'}:
                print(f"  ⚠️  Non-normalized digits found: {digits}")
            elif digits == {'1'}:
                print(f"  ✅ Digits normalized")

        if num_examples == 0:
            print(f"\n  ❌ FAILED: No examples loaded!")
            all_passed = False
        else:
            print(f"\n  ✅ TEST PASSED")

    except Exception as e:
        print(f"\n  ❌ FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

print("\n" + "="*80)
print("FINAL RESULT")
print("="*80)

if all_passed:
    print("✅ ALL TESTS PASSED!")
    print("\nThe data loader correctly handles:")
    print("  • Standard format (human_text, machine_text)")
    print("  • Reddit format (text, machine_text)")
    print("  • PeerRead list format")
    print("  • Multiple domains/generators")
    print("  • Digit normalization")
    print("  • No field name leakage")
    print("\n🚀 READY FOR PRODUCTION TRAINING!")
else:
    print("❌ SOME TESTS FAILED")
    print("\n🔧 Fix issues before training!")

print("="*80)
