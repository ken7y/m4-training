#!/usr/bin/env python3
"""
Test M4 data loading to ensure:
1. Both human_text and machine_text are loaded correctly
2. Labels are correct (0=human, 1=machine)
3. No field names leak into training data
"""

import json
import re

def normalize_digits(text):
    return re.sub(r'\d', '1', text)

def load_m4_data_CORRECTED(data_file, max_samples=5):
    """
    Correct M4 data loader.
    Each JSON line has BOTH human_text and machine_text.
    We create TWO examples from each line.
    """
    data = []

    print(f"Loading from: {data_file}\n")
    print("="*80)

    with open(data_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break

            item = json.loads(line.strip())

            # Extract both texts
            human_text = item.get('human_text', '')
            machine_text = item.get('machine_text', '')

            if not human_text or not machine_text:
                print(f"⚠️  Line {i+1}: Missing text fields!")
                continue

            # Create TWO training examples
            # Example 1: Human text with label=0
            data.append({
                'text': human_text,
                'label': 0,  # 0 = human
                'generator': 'human',
                'domain': item.get('source', 'unknown'),
                'id': f"{item.get('source_ID', i)}_human"
            })

            # Example 2: Machine text with label=1
            data.append({
                'text': machine_text,
                'label': 1,  # 1 = machine
                'generator': item.get('model', 'unknown'),
                'domain': item.get('source', 'unknown'),
                'id': f"{item.get('source_ID', i)}_machine"
            })

            print(f"\n📄 Line {i+1} → Created 2 examples:")
            print(f"  Example {len(data)-1}:")
            print(f"    Label: 0 (HUMAN)")
            print(f"    Length: {len(human_text)} chars")
            print(f"    Preview: {human_text[:150].strip()}...")
            print(f"    No 'human_text:' prefix: {('human_text:' not in human_text[:200])}")

            print(f"  Example {len(data)}:")
            print(f"    Label: 1 (MACHINE)")
            print(f"    Generator: {item.get('model', 'unknown')}")
            print(f"    Length: {len(machine_text)} chars")
            print(f"    Preview: {machine_text[:150].strip()}...")
            print(f"    No 'machine_text:' prefix: {('machine_text:' not in machine_text[:200])}")

            print("-"*80)

    return data

def test_old_loader(data_file, max_samples=5):
    """Test the OLD (BROKEN) loader"""
    print("\n🚨 TESTING OLD (BROKEN) LOADER:\n")
    data = []

    with open(data_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break

            item = json.loads(line.strip())
            # OLD CODE (looking for 'text' field that doesn't exist)
            text = item.get('text', item.get('content', ''))

            if not text:
                print(f"❌ Line {i+1}: No 'text' or 'content' field found!")
            else:
                data.append({'text': text})

    print(f"\n❌ OLD LOADER RESULT: {len(data)} examples loaded (should be 10, got {len(data)})")
    return data

if __name__ == "__main__":
    data_file = "M4/data/arxiv_chatGPT.jsonl"

    # Test old broken loader
    old_data = test_old_loader(data_file, max_samples=5)

    # Test new correct loader
    print("\n" + "="*80)
    print("✅ TESTING NEW (CORRECT) LOADER:\n")
    new_data = load_m4_data_CORRECTED(data_file, max_samples=5)

    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY:")
    print(f"  OLD loader: {len(old_data)} examples (BROKEN! ❌)")
    print(f"  NEW loader: {len(new_data)} examples (CORRECT! ✅)")
    print(f"  Expected: 10 examples (5 lines × 2 texts each)")

    if len(new_data) == 10:
        print("\n✅ CORRECT: Each JSON line creates 2 examples (human + machine)")

        # Verify labels
        human_count = sum(1 for d in new_data if d['label'] == 0)
        machine_count = sum(1 for d in new_data if d['label'] == 1)
        print(f"\n  Human examples (label=0): {human_count}")
        print(f"  Machine examples (label=1): {machine_count}")

        if human_count == machine_count == 5:
            print("\n✅ Labels are balanced and correct!")

        # Check for field name leakage
        has_leakage = any('human_text:' in d['text'] or 'machine_text:' in d['text'] for d in new_data)
        if not has_leakage:
            print("✅ No field names in training data!")
        else:
            print("❌ WARNING: Field names found in data!")
    else:
        print("\n❌ ERROR: Incorrect number of examples!")

    print("="*80)
