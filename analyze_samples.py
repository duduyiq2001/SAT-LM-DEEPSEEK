import json
import re
import sys
import difflib
from collections import defaultdict

# Load test data
test_file = "data/arlsat_test.json"
with open(test_file, 'r') as f:
    test_data = json.load(f)

# Load translation file
trans_file = "misc/multisgate-trans-arlsat--eng--test0-230--sigsigz3--stsetupsatlm--3--numsamp1--temp0.0--stysigtpl--modelgpt-4.1--predictions.json"
with open(trans_file, 'r') as f:
    trans_data = json.load(f)

print(f"Loaded {len(test_data)} test samples and {len(trans_data)} translation items")

# Load mapping data
mapping_file = "misc/translation-mapping.json"
with open(mapping_file, 'r') as f:
    mapping_data = json.load(f)

# Analyze matches
match_results = mapping_data['match_results']
matched_items = [m for m in match_results if m['status'] == 'matched']

# Count how many matches each test index has
test_idx_counts = defaultdict(int)
for match in matched_items:
    test_idx_counts[match['test_idx']] += 1

print(f"Test indices with matches: {len(test_idx_counts)}")
print(f"Test indices with multiple matches: {sum(1 for count in test_idx_counts.values() if count > 1)}")

# Find most duplicated test indices
most_duplicated = sorted([(idx, count) for idx, count in test_idx_counts.items()], 
                         key=lambda x: x[1], reverse=True)
print("\nMost duplicated test indices:")
for idx, count in most_duplicated[:5]:
    print(f"Test index {idx}: {count} matches")

# Extract and print the first 100 chars of the test sample for these indices
print("\nExample duplicates analysis:")
for idx, count in most_duplicated[:3]:
    print(f"\nTest index {idx} ({count} matches):")
    
    # Print test data
    test_item = test_data[idx]
    context_excerpt = test_item.get('context', '')[:150]
    print(f"Test context: {context_excerpt}...")
    
    # Find all translations matched to this test index
    matching_trans = [m['trans_idx'] for m in matched_items if m['test_idx'] == idx]
    
    # Print the first 100 chars of each matching translation's text
    for i, trans_idx in enumerate(matching_trans[:3]):  # Show first 3
        item = trans_data[trans_idx]
        if item and item[0] and item[0][0]:
            prompt = item[0][0].get('prompt', '')
            
            # Extract test sample from prompt
            pattern = r'"""(.*?)"""'
            matches = re.findall(pattern, prompt, re.DOTALL)
            test_sample = matches[-1].strip() if matches else ""
            
            print(f"Match {i+1} (trans_idx {trans_idx}):")
            print(f"  Sample excerpt: {test_sample[:150]}...")

# Analyze test data coverage
matched_indices = set(test_idx_counts.keys())
all_indices = set(range(len(test_data)))
unmatched_indices = all_indices - matched_indices

print(f"\nTest data coverage:")
print(f"Matched: {len(matched_indices)}/{len(test_data)} ({len(matched_indices)/len(test_data)*100:.1f}%)")
print(f"Unmatched: {len(unmatched_indices)}/{len(test_data)} ({len(unmatched_indices)/len(test_data)*100:.1f}%)")

# Print some unmatched examples for analysis
print("\nExample unmatched test items:")
for idx in list(unmatched_indices)[:3]:
    test_item = test_data[idx]
    context_excerpt = test_item.get('context', '')[:100]
    print(f"\nTest index {idx}:")
    print(f"Context: {context_excerpt}...") 