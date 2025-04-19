#!/usr/bin/env python3
"""
Quick test script to check if we can extract test samples correctly from
both signature and translation files.
"""

import json
import re
import sys

def extract_test_sample(text):
    """Extract the test sample (context and question) from text"""
    # Look for patterns that indicate the test problem statement
    context_match = re.search(r'"""(.*?)"""', text, re.DOTALL)
    if context_match:
        return context_match.group(1).strip()
    
    # Try another common pattern
    context_match = re.search(r'Z3 CODE GENERATION TASK.*?"""(.*?)"""', text, re.DOTALL)
    if context_match:
        return context_match.group(1).strip()
    
    # Look for problem context and question pattern
    context_match = re.search(r'(?:Given|Context:|Problem:)(.*?)(?:Question:|What)', text, re.DOTALL)
    if context_match:
        return context_match.group(1).strip()
    
    # Fall back to first paragraph if nothing else works
    paragraphs = text.split('\n\n')
    if paragraphs:
        return paragraphs[0].strip()
    
    return text[:200].strip()  # Return first 200 chars as last resort

def normalize_text(text):
    """Normalize text for comparison by removing extra spaces and lowercase"""
    return re.sub(r'\s+', ' ', text).lower().strip()

def read_json(filepath):
    """Read a JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def main():
    # Files to test
    sig_file = "misc/multisgate-sig-arlsat--test0-230--manualsigz3--numsamp1--temp0.0--stysigtpl--modelgpt-4.1--predictions.json"
    trans_file = "misc/multisgate-trans-arlsat--eng--test0-230--sigsigz3--stsetupsatlm--3--numsamp1--temp0.0--stysigtpl--modelgpt-4.1--predictions.json"
    test_data_file = "data/arlsat_test.json"
    
    # Load files
    sig_results = read_json(sig_file)
    trans_results = read_json(trans_file)
    test_data = read_json(test_data_file)
    
    # Check a sample from each
    num_samples = 5
    
    # Process test data for matching
    test_samples = []
    for i, item in enumerate(test_data[:num_samples]):
        if "context" in item:
            sample = item["context"]
            if "question" in item:
                sample += " " + item["question"]
        else:
            sample = item.get("id", "")
        test_samples.append((i, normalize_text(sample)))
    
    print(f"Testing extraction from {num_samples} signature samples...")
    for i in range(min(num_samples, len(sig_results))):
        example = sig_results[i]
        if not example:
            print(f"Example {i}: Empty")
            continue
        
        text = example[0][0]['text'] if example[0] else ""
        if not text:
            print(f"Example {i}: No text")
            continue
        
        sample = extract_test_sample(text)
        normalized = normalize_text(sample)
        
        # Find best match in test data
        best_match = -1
        best_score = 0
        for test_idx, test_sample in test_samples:
            import difflib
            score = difflib.SequenceMatcher(None, normalized[:200], test_sample[:200]).ratio()
            if score > best_score:
                best_score = score
                best_match = test_idx
        
        print(f"Example {i}: Extracted {len(sample)} chars, best match with test {best_match} (score: {best_score:.2f})")
        print(f"Sample: {sample[:100]}...")
    
    print("\nTesting extraction from {num_samples} translation samples...")
    for i in range(min(num_samples, len(trans_results))):
        example = trans_results[i]
        if not example:
            print(f"Example {i}: Empty")
            continue
        
        text = example[0][0]['text'] if example[0] else ""
        if not text:
            print(f"Example {i}: No text")
            continue
        
        sample = extract_test_sample(text)
        normalized = normalize_text(sample)
        
        # Find best match in test data
        best_match = -1
        best_score = 0
        for test_idx, test_sample in test_samples:
            import difflib
            score = difflib.SequenceMatcher(None, normalized[:200], test_sample[:200]).ratio()
            if score > best_score:
                best_score = score
                best_match = test_idx
        
        print(f"Example {i}: Extracted {len(sample)} chars, best match with test {best_match} (score: {best_score:.2f})")
        print(f"Sample: {sample[:100]}...")

if __name__ == "__main__":
    main() 