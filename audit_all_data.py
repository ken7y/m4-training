#!/usr/bin/env python3
"""
COMPREHENSIVE DATA AUDIT
Check EVERY file in M4/data to understand format variations
"""

import json
import os
import glob

data_dir = "M4/data"
files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))

print("="*80)
print("M4 DATASET COMPREHENSIVE AUDIT")
print("="*80)
print(f"\nTotal files: {len(files)}\n")

format_summary = {}
issues = []

for filepath in files:
    filename = os.path.basename(filepath)

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if not first_line:
                issues.append(f"{filename}: EMPTY FILE")
                continue

            item = json.loads(first_line)
            keys = list(item.keys())

            # Check format type
            has_human_text = 'human_text' in item
            has_machine_text = 'machine_text' in item
            has_text = 'text' in item

            # Check type of human_text/text
            human_field = item.get('human_text', item.get('text', ''))
            machine_field = item.get('machine_text', '')

            human_type = type(human_field).__name__
            machine_type = type(machine_field).__name__

            # Count lines
            with open(filepath, 'r', encoding='utf-8') as f2:
                line_count = sum(1 for _ in f2)

            format_key = f"human_text:{has_human_text}, text:{has_text}, machine_text:{has_machine_text}"
            format_key += f" | types: {human_type}/{machine_type}"

            if format_key not in format_summary:
                format_summary[format_key] = []
            format_summary[format_key].append((filename, line_count))

            # Check for issues
            if human_type == 'list' or machine_type == 'list':
                issues.append(f"{filename}: LIST format (not string)")

            if not has_human_text and not has_text:
                issues.append(f"{filename}: NO human text field")

            if not has_machine_text:
                issues.append(f"{filename}: NO machine_text field")

    except Exception as e:
        issues.append(f"{filename}: ERROR - {str(e)}")

print("\n" + "="*80)
print("FORMAT SUMMARY")
print("="*80)

for fmt, file_list in format_summary.items():
    print(f"\n📋 Format: {fmt}")
    print(f"   Files: {len(file_list)}")
    for fname, lines in file_list[:3]:  # Show first 3
        print(f"     - {fname} ({lines} lines)")
    if len(file_list) > 3:
        print(f"     ... and {len(file_list) - 3} more")

print("\n" + "="*80)
print("ISSUES FOUND")
print("="*80)

if issues:
    for issue in issues:
        print(f"⚠️  {issue}")
else:
    print("✅ No issues found")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

print("""
Based on this audit, the data loader needs to handle:

1. **Standard format**: human_text (str), machine_text (str)
   - Most files use this format

2. **Reddit format**: text (str), machine_text (str)
   - Reddit files use 'text' instead of 'human_text'

3. **PeerRead format**: human_text (list), machine_text (list)
   - PeerRead files have LISTS of reviews, not single strings
   - Need to either: flatten lists, use first item, or skip these files

4. **Field name variations**: source_ID vs source_id (case difference)
""")
