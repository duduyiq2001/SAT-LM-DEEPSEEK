#!/usr/bin/env python3
"""
Script to evaluate the correctness of the translation results.

This script:
1. Loads the translation results
2. Extracts Z3 code from each result
3. Attempts to execute the code to check for syntax errors
4. For each example, evaluates correctness based on executing the code
5. Reports overall accuracy and stats
"""

import os
import sys
import json
import re
import argparse
import tempfile
import subprocess
from collections import defaultdict
import importlib.util
import traceback
from tqdm import tqdm

# Import the existing Z3 execution function
sys.path.append('.')
from prog_solver.z3_utils import execute_z3_test

def read_json(filepath):
    """Read a JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_z3_code(text):
    """Extract Z3 code from the response text"""
    # Try multiple patterns to extract code with python marker
    code_blocks = re.findall(r'```(?:python|python3|z3)?\s*(.*?)```', text, re.DOTALL)
    
    if code_blocks:
        # Use the first code block
        code = code_blocks[0].strip()
    else:
        # Look for code between triple backticks without language marker
        code_blocks = re.findall(r'```\s*(.*?)```', text, re.DOTALL)
        if code_blocks:
            code = code_blocks[0].strip()
        else:
            # If no code blocks, try to extract code starting with 'from z3 import'
            import_match = re.search(r'(from\s+z3\s+import.*?)(?:\n\n\n|$)', text, re.DOTALL)
            if import_match:
                code = import_match.group(1).strip()
            else:
                # Last resort: just use the entire text, but clean it
                code = text.strip()
                # Remove markdown-style formatting if present
                code = re.sub(r'^#+\s+.*$', '', code, flags=re.MULTILINE)
    
    # Ensure code has Z3 imports
    if 'from z3 import' not in code and 'import z3' not in code:
        code = 'from z3 import *\n' + code
    
    # Make sure there's a check_valid function
    if 'def check_valid' not in code:
        # Try to find any function definition
        func_match = re.search(r'def\s+(\w+)\s*\(', code)
        if func_match:
            main_func = func_match.group(1)
            
            # Check if the main function has a return statement
            func_body_match = re.search(r'def\s+' + main_func + r'\s*\([^)]*\):(.+?)(?=\ndef|\Z)', code, re.DOTALL)
            if func_body_match and 'return' not in func_body_match.group(1):
                # Add code to print the final answer if there's no return
                main_func_code = """
if __name__ == "__main__":
    result = {}()
    if isinstance(result, tuple):
        for i, res in enumerate(result):
            if str(res) == 'sat':
                print(chr(65 + i))  # A, B, C, D, E
                break
    elif result is not None:
        print(result)
""".format(main_func)
                code += main_func_code
            else:
                # Just call the function
                code += f'\n\nif __name__ == "__main__":\n    print({main_func}())'
        else:
            # Wrap code in check_valid function if no function exists
            indented_code = '\n'.join(['    ' + line for line in code.split('\n')])
            code = f'def check_valid():\n{indented_code}\n\nif __name__ == "__main__":\n    result = check_valid()\n    if result is not None:\n        print(result)'
    else:
        # Make sure the function is called and result is printed
        if 'if __name__ == "__main__"' not in code:
            # Check if check_valid returns anything
            check_valid_body = re.search(r'def\s+check_valid\s*\([^)]*\):(.+?)(?=\ndef|\Z)', code, re.DOTALL)
            if check_valid_body and 'return' in check_valid_body.group(1):
                code += '\n\nif __name__ == "__main__":\n    result = check_valid()\n    if isinstance(result, tuple):\n        for i, res in enumerate(result):\n            if str(res) == "sat":\n                print(chr(65 + i))\n                break\n    elif result is not None:\n        print(result)'
            else:
                code += '\n\nif __name__ == "__main__":\n    check_valid()'
    
    # Add code to handle common patterns of answer extraction
    if 'solver.check()' in code and 'print' not in code:
        # Add code to print the result of solver.check()
        code = code.replace('solver.check()', 'result = solver.check()\nprint(result)\nreturn result')
    
    # Handle multiple solvers (sA, sB, sC, etc.)
    solvers_match = re.findall(r's([A-E])\s*=\s*[^;]+check\(\)', code)
    if solvers_match and 'print' not in code:
        # Add code to the end to print which solver is satisfiable
        code_lines = code.split('\n')
        insert_pos = -1
        for i, line in enumerate(reversed(code_lines)):
            if line.strip() and not line.strip().startswith('#'):
                insert_pos = len(code_lines) - i
                break
        
        if insert_pos > 0:
            print_code = '\n    # Print the result\n'
            for solver_letter in 'ABCDE':
                if solver_letter.lower() in [s.lower() for s in solvers_match]:
                    print_code += f'    if s{solver_letter}.check() == sat:\n        print("{solver_letter}")\n'
            
            code_lines.insert(insert_pos, print_code)
            code = '\n'.join(code_lines)
    
    return code

def is_z3_installed():
    """Check if Z3 is installed"""
    try:
        import z3
        return True
    except ImportError:
        return False

def check_code_syntax(code):
    """Check if the code has valid syntax"""
    try:
        compile(code, '<string>', 'exec')
        return True, None
    except SyntaxError as e:
        return False, str(e)

def parse_z3_output(output):
    """Parse the output of a Z3 solver execution to extract the answer"""
    # Look for common patterns in Z3 output for SAT problems
    
    # Pattern 1: Explicit answer statement
    answer_match = re.search(r'(?:Answer|The answer)(?:\s+is)?[\s:]*\(?([A-E])\)?', output, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()
    
    # Pattern 2: Direct choice indication (A), (B), etc.
    option_matches = re.findall(r'[\(\[]\s*([A-E])\s*[\)\]]', output)
    if option_matches:
        return option_matches[-1].upper()
    
    # Pattern 3: "Option X is correct/true"
    option_correct = re.search(r'[Oo]ption\s+([A-E])(?:\s+is)?\s+(?:correct|true)', output)
    if option_correct:
        return option_correct.group(1).upper()
    
    # Pattern 4: Look for 'sat' followed by a letter (common Z3 output pattern)
    sat_match = re.search(r'sat.*?([A-E])', output, re.IGNORECASE)
    if sat_match:
        return sat_match.group(1).upper()
        
    # Pattern 5: Look for any letter followed by 'sat' (another common pattern)
    letter_sat_match = re.search(r'([A-E]).*?sat', output, re.IGNORECASE)
    if letter_sat_match:
        return letter_sat_match.group(1).upper()
    
    # Pattern 6: Check for multiple 'sat'/'unsat' checks with letters
    sat_checks = re.findall(r'([A-E]).*?(sat|unsat)', output, re.IGNORECASE)
    if sat_checks:
        # Find the letter that corresponds to 'sat'
        for letter, result in sat_checks:
            if result.lower() == 'sat':
                return letter.upper()
        
        # If all are unsat except one, the remaining one is the answer
        found_letters = set(letter.upper() for letter, _ in sat_checks)
        all_letters = set('ABCDE')
        if len(found_letters) == 4:  # If 4 were checked, the 5th is the answer
            possible_answer = all_letters - found_letters
            if len(possible_answer) == 1:
                return possible_answer.pop()
    
    # Pattern 7: Look for True/False checks with letters
    true_checks = re.findall(r'([A-E]).*?(True|False)', output, re.IGNORECASE)
    if true_checks:
        # Find the letter that corresponds to 'True'
        for letter, result in true_checks:
            if result.lower() == 'true':
                return letter.upper()
    
    # Pattern 8: look for statements like "print('A')" or "print(A)" at the end of output
    print_match = re.search(r'(?:^|\n)([A-E])(?:\s*$)', output)
    if print_match:
        return print_match.group(1).upper()
    
    # Pattern 9: Look for index results that map to letters (e.g., [0] -> A, [1] -> B)
    index_match = re.search(r'\[\s*(\d)\s*\]', output)
    if index_match:
        index = int(index_match.group(1))
        if 0 <= index <= 4:
            return chr(ord('A') + index)
    
    # Pattern 10: Look for capital letters that might be the answer
    capital_letters = re.findall(r'(?<![a-zA-Z])([A-E])(?![a-zA-Z])', output)
    if capital_letters:
        # Count occurrences of each letter
        from collections import Counter
        letter_counts = Counter(capital_letters)
        # Get the most frequent letter
        most_common = letter_counts.most_common(1)
        if most_common and most_common[0][1] > 1:  # If it appears more than once
            return most_common[0][0].upper()
    
    # If nothing was found, return None
    return None

def extract_test_sample(text):
    """Extract the test sample (context and question) from translation text"""
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

def find_best_match(sample, test_data):
    """Find the best matching test data for a given sample"""
    best_match_idx = -1
    best_match_score = 0
    
    sample_normalized = normalize_text(sample)
    
    for i, test_item in enumerate(test_data):
        # Extract problem text from test data
        if "context" in test_item:
            test_sample = test_item["context"]
            if "question" in test_item:
                test_sample += " " + test_item["question"]
        else:
            # Extract from first sentence or just use id if nothing else
            test_sample = test_item.get("id", "")
        
        test_normalized = normalize_text(test_sample)
        
        # Calculate similarity score using character sequence matching
        import difflib
        similarity = difflib.SequenceMatcher(None, sample_normalized[:200], test_normalized[:200]).ratio()
        
        if similarity > best_match_score:
            best_match_score = similarity
            best_match_idx = i
    
    # Only return a match if it's reasonably similar
    if best_match_score > 0.6:
        return best_match_idx, best_match_score
    return -1, 0

def evaluate_translation(trans_file, test_data_file=None, matching_threshold=0.6):
    """Evaluate translation results"""
    print(f"Loading translation results from: {trans_file}")
    trans_results = read_json(trans_file)
    
    ground_truth = None
    test_data = None
    test_ids = []
    if test_data_file:
        print(f"Loading test data from: {test_data_file}")
        test_data = read_json(test_data_file)
        ground_truth = [ex.get("label", None) for ex in test_data]
        test_ids = [ex.get("id", f"sample_{i}") for i, ex in enumerate(test_data)]
        print(f"Using matching threshold: {matching_threshold}")
    
    # Check if Z3 is installed
    if not is_z3_installed():
        print("ERROR: Z3 is not installed. Please install it with 'pip install z3-solver'")
        return
    
    # Ensure the tmp directory exists for the Z3 execution function
    os.makedirs("tmp", exist_ok=True)
    
    # Statistics
    stats = {
        "total": len(trans_results),
        "syntax_errors": 0,
        "execution_errors": 0,
        "no_answer": 0,
        "correct": 0,
        "incorrect": 0,
        "unknown": 0,
        "match_failed": 0
    }
    
    results = []
    
    print(f"Evaluating {len(trans_results)} translation results...")
    print(f"Warning: Translation results length {len(trans_results)} vs test data length {len(test_data) if test_data else 'N/A'}")
    
    for i, example in enumerate(tqdm(trans_results)):
        result = {
            "index": i,
            "syntax_valid": False,
            "execution_success": False,
            "predicted_answer": None,
            "correct": None,
            "error": None,
            "match_info": None
        }
        
        # Skip empty examples
        if not example:
            stats["unknown"] += 1
            results.append(result)
            continue
        
        # Use the first response for each example
        response = example[0][0]["text"] if example[0] else ""
        if not response:
            stats["unknown"] += 1
            results.append(result)
            continue
        
        # Use the same index from test data
        matched_test_idx = i  # Direct index matching
        test_id = test_ids[i] if i < len(test_ids) else None
        
        # Record match info for diagnostics
        result["match_info"] = {
            "original_idx": i,
            "matched_idx": matched_test_idx,
            "test_id": test_id
        }
        
        # Extract Z3 code
        code = extract_z3_code(response)
        
        # Check syntax
        syntax_valid, syntax_error = check_code_syntax(code)
        result["syntax_valid"] = syntax_valid
        
        if not syntax_valid:
            stats["syntax_errors"] += 1
            result["error"] = f"Syntax error: {syntax_error}"
            results.append(result)
            continue
        
        # Execute code using the existing Z3 execution function
        execution_success, output = execute_z3_test(code, timeout=15.0)
        result["execution_success"] = execution_success
        
        if not execution_success:
            stats["execution_errors"] += 1
            result["error"] = f"Execution error: {output}"
            results.append(result)
            continue
        
        # Parse output to get answer
        answer = parse_z3_output(output)
        result["predicted_answer"] = answer
        
        if not answer:
            stats["no_answer"] += 1
            result["error"] = "Could not determine answer from output"
            results.append(result)
            continue
        
        # Convert letter answer to index (A=0, B=1, etc.)
        answer_idx = ord(answer) - ord('A')
        result["predicted_index"] = answer_idx
        
        # Compare with ground truth if available
        if ground_truth and matched_test_idx < len(ground_truth) and ground_truth[matched_test_idx] is not None:
            expected_idx = ground_truth[matched_test_idx]
            result["correct"] = (answer_idx == expected_idx)
            result["expected_index"] = expected_idx
            result["expected_answer"] = chr(ord('A') + expected_idx)
            
            if result["correct"]:
                stats["correct"] += 1
            else:
                stats["incorrect"] += 1
        else:
            stats["unknown"] += 1
        
        results.append(result)
    
    # Calculate accuracy if ground truth is available
    if ground_truth:
        accuracy = stats["correct"] / (stats["correct"] + stats["incorrect"]) if (stats["correct"] + stats["incorrect"]) > 0 else 0
        stats["accuracy"] = accuracy
    
    return stats, results

def main():
    parser = argparse.ArgumentParser(description='Evaluate translation results')
    parser.add_argument('--trans-file', required=True, help='Path to translation results JSON')
    parser.add_argument('--test-data', help='Path to test data JSON with ground truth')
    parser.add_argument('--output', help='Path to save evaluation results')
    parser.add_argument('--diagnostics', help='Path to save detailed diagnostics', default='eval_diagnostics.json')
    parser.add_argument('--matching-threshold', type=float, default=0.6, help='Threshold for text matching (0.0-1.0)')
    args = parser.parse_args()
    
    stats, results = evaluate_translation(args.trans_file, args.test_data, args.matching_threshold)
    
    # Print matching statistics
    match_failed = stats.get("match_failed", 0)
    print(f"\nMatching Statistics:")
    print(f"Examples with failed matching: {match_failed} ({match_failed/stats['total']*100:.1f}%)")
    
    # Calculate match score statistics if available
    match_scores = [r.get("match_info", {}).get("match_score", 0) for r in results if "match_info" in r]
    if match_scores:
        avg_match_score = sum(match_scores) / len(match_scores)
        print(f"Average match score: {avg_match_score:.2f}")
        
        # Count matches by score range
        excellent_matches = sum(1 for s in match_scores if s >= 0.9)
        good_matches = sum(1 for s in match_scores if 0.75 <= s < 0.9)
        fair_matches = sum(1 for s in match_scores if 0.6 <= s < 0.75)
        
        print(f"Excellent matches (≥0.9): {excellent_matches} ({excellent_matches/len(match_scores)*100:.1f}%)")
        print(f"Good matches (0.75-0.9): {good_matches} ({good_matches/len(match_scores)*100:.1f}%)")
        print(f"Fair matches (0.6-0.75): {fair_matches} ({fair_matches/len(match_scores)*100:.1f}%)")
    
    # Save detailed diagnostics
    incorrect_cases = [r for r in results if r.get('correct') is False]
    if incorrect_cases:
        diagnostics = {
            "incorrect_count": len(incorrect_cases),
            "match_failed_count": match_failed,
            "details": incorrect_cases
        }
        
        # Add test data for reference
        if args.test_data:
            test_data = read_json(args.test_data)
            for case in diagnostics["details"]:
                idx = case.get("match_info", {}).get("matched_idx", -1)
                if idx >= 0 and idx < len(test_data):
                    case["test_data"] = test_data[idx]
        
        with open(args.diagnostics, 'w') as f:
            json.dump(diagnostics, f, indent=2)
        print(f"\nDetailed diagnostics for {len(incorrect_cases)} incorrect cases saved to {args.diagnostics}")
    
    # Print statistics
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total examples: {stats['total']}")
    print(f"Syntax errors: {stats['syntax_errors']} ({stats['syntax_errors']/stats['total']*100:.1f}%)")
    print(f"Execution errors: {stats['execution_errors']} ({stats['execution_errors']/stats['total']*100:.1f}%)")
    print(f"No answer extracted: {stats['no_answer']} ({stats['no_answer']/stats['total']*100:.1f}%)")
    
    if "accuracy" in stats:
        print(f"Correct answers: {stats['correct']} ({stats['correct']/stats['total']*100:.1f}%)")
        print(f"Incorrect answers: {stats['incorrect']} ({stats['incorrect']/stats['total']*100:.1f}%)")
        print(f"Unknown (no ground truth): {stats['unknown']} ({stats['unknown']/stats['total']*100:.1f}%)")
        print(f"\nAccuracy: {stats['accuracy']*100:.2f}%")
        
        # Print example of an incorrect case for debugging
        if incorrect_cases:
            print("\nExample of incorrect case:")
            case = incorrect_cases[0]
            idx = case["index"]
            match_idx = case.get("match_info", {}).get("matched_idx", idx)
            match_score = case.get("match_info", {}).get("match_score", 0)
            
            print(f"Index: {idx}, Matched to test sample: {match_idx} (score: {match_score:.2f})")
            print(f"Expected: {case.get('expected_answer')}, Got: {case.get('predicted_answer')}")
            
            if "error" in case:
                print(f"Error: {case['error']}")
    
    # Save results if output path is specified
    if args.output:
        output_data = {
            "stats": stats,
            "results": results
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main() 