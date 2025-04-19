#!/usr/bin/env python3
"""
Simplified pipeline for SAT-LM:
1. Signature generation
2. Translation
3. Evaluation

This version processes one problem at a time and prints detailed outputs
without using the caching mechanisms.
"""

import os
import sys
import json
import logging
import argparse
import tempfile
import hashlib
from pathlib import Path

# Add parent directory to path
sys.path.append('.')

# Import the evaluator functions only
from evaluate_translation import extract_z3_code, execute_z3_test, parse_z3_output
from api_utils import gpt_safe_completion

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def satlm_encode_question(ex):
    """Encodes a question in SATLM style
    
    Args:
        ex (dict): Question data containing context, question, and choices
        
    Returns:
        str: Formatted question text with context, question, and choices
    """
    context = ex.get("context", "")
    question = ex.get("question", "")
    if not question.endswith("."):
        question = question + "."
    
    # Identifiers for multiple-choice options
    options = ["(A)", "(B)", "(C)", "(D)", "(E)"]

    choices = []
    for i, choice in enumerate(ex.get("choices", [])):
        if i < len(options):
            choices.append("{} {}".format(options[i], choice))
    choices_str = "\n".join(choices)

    result = f"Question: {context}\n\n{question}\n\n{choices_str}"
    return result

def get_cache_filename(args, stage, prompt, index):
    """Generate a unique filename for caching results
    
    Args:
        args: Command line arguments
        stage: 'signature' or 'translation'
        prompt: The prompt being cached
        index: Index of the example
        
    Returns:
        str: Cache filename
    """
    # Create a hash of the prompt to ensure unique filenames
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
    
    filename = f"misc/simple-{stage}-{args.task}--{args.split}{index}--{args.model}--{prompt_hash}.json"
    return filename

def load_from_cache(cache_file):
    """Load cached results if they exist
    
    Args:
        cache_file: Path to the cache file
        
    Returns:
        dict or None: Cached results or None if not found
    """
    if not args.use_cache:
        return None
        
    if os.path.exists(cache_file):
        logger.info(f"Loading from cache: {cache_file}")
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    return None

def save_to_cache(cache_file, response):
    """Save results to cache
    
    Args:
        cache_file: Path to the cache file
        response: API response to cache
    """
    if not args.use_cache:
        return
        
    logger.info(f"Saving to cache: {cache_file}")
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        
        with open(cache_file, 'w') as f:
            json.dump(response, f)
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")

def setup_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run simplified SAT-LM pipeline')
    
    # Dataset selection
    parser.add_argument('--task', type=str, default='arlsat', choices=['arlsat', 'clutrr'], 
                        help='Task name (default: arlsat)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'],
                        help='Data split (default: test)')
    parser.add_argument('--start-idx', type=int, default=0, 
                        help='Starting index for test samples (default: 0)')
    parser.add_argument('--num-samples', type=int, default=1, 
                        help='Number of test samples to process (default: 1)')
    
    # Model selection
    parser.add_argument('--model', type=str, default='gpt-4.1', 
                        help='Model to use (default: gpt-4.1)')
    parser.add_argument('--temperature', type=float, default=0.0, 
                        help='Temperature for sampling (default: 0.0)')
    
    # Caching options
    parser.add_argument('--use-cache', action='store_true',
                        help='Use cached results when available (default: False)')
    parser.add_argument('--cache-dir', type=str, default='misc',
                        help='Directory for caching results (default: misc)')
    
    args = parser.parse_args()
    return args

def load_test_data(task, split, start_idx, num_samples):
    """Load test data for the specified task"""
    data_path = f"data/{task}_{split}.json"
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)
        
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Extract the requested subset of data
    end_idx = min(start_idx + num_samples, len(data))
    subset = data[start_idx:end_idx]
    
    logger.info(f"Loaded {len(subset)} test samples from {data_path} (indices {start_idx}-{end_idx-1})")
    return subset

def generate_signature_prompt(test_sample):
    """Generate a signature prompt for the given test sample"""
    # Use the SATLM encoding function to format the question
    formatted_question = satlm_encode_question(test_sample)
    
    prompt = f"""You are generating a Z3 problem signature to solve a logical reasoning task. 
The signature should define the variables and constraints of the problem in a mathematical format.

## Example:

From z3 import *

def check_valid():
    solver = Solver()
    
    # Define variables
    # ...
    
    # Define constraints
    # ...
    
    # Check if constraints are satisfiable
    if solver.check() == sat:
        model = solver.model()
        # Extract solution
        # ...
        print("The answer is X")
    else:
        print("No solution found")

check_valid()

## Now, create a Z3 signature for the following problem:

{formatted_question}

Start your Z3 signature with "from z3 import *":
"""
    return prompt

def generate_translation_prompt(test_sample, signature):
    """Generate a translation prompt for the given test sample and signature"""
    # Use the SATLM encoding function to format the question
    formatted_question = satlm_encode_question(test_sample)
    
    prompt = f"""You are translating a problem signature into executable Z3 code that will solve a logical reasoning task.

## Problem:

{formatted_question}

## Signature:
{signature}

## Instructions:
1. Implement the signature as executable Z3 code
2. Make sure to include the check_valid() function
3. Your code should print the answer clearly as 'The answer is X' where X is A, B, C, D, or E
4. Include proper error handling and ensure the code is well-structured

## Executable Z3 code:
"""
    return prompt

def process_single_example(args, test_sample, example_idx):
    """Process a single example through the full pipeline"""
    print("\n" + "=" * 80)
    print(f"PROCESSING EXAMPLE {example_idx}")
    print("=" * 80)
    
    # Display problem
    print(f"\nCONTEXT: {test_sample.get('context', '')}")
    print(f"QUESTION: {test_sample.get('question', '')}")
    
    # Display all choices
    choices = test_sample.get('choices', [])
    if choices:
        print("\nCHOICES:")
        for i, choice in enumerate(choices):
            print(f"  {chr(ord('A') + i)}. {choice}")
    
    if test_sample.get('label') is not None:
        print(f"\nEXPECTED ANSWER: {chr(ord('A') + test_sample['label'])}")
    print("=" * 80)
    
    # Step 1: Generate Signature
    print("\nSTEP 1: SIGNATURE GENERATION")
    print("=" * 50)
    
    signature_prompt = generate_signature_prompt(test_sample)
    
    # Check cache for signature
    signature_cache_file = get_cache_filename(args, "signature", signature_prompt, args.start_idx + example_idx)
    signature_responses = load_from_cache(signature_cache_file)
    
    if signature_responses:
        print(f"Using cached signature from {signature_cache_file}")
    else:
        print("Generating signature...")
        signature_responses = gpt_safe_completion(
            [signature_prompt], 
            temperature=args.temperature, 
            max_tokens=2048, 
            stop_token=None,
            model=args.model
        )
        # Save to cache
        save_to_cache(signature_cache_file, signature_responses)
    
    # Extract the signature text from the API response
    if not signature_responses:
        print("SIGNATURE GENERATION FAILED - No response received")
        return False
    
    # Debug info about response structure
    signature_text = ""
    if isinstance(signature_responses, dict) and 'choices' in signature_responses:
        # Handle API response format for chat models
        print("Extracting signature from chat completion response...")
        choices = signature_responses.get('choices', [])
        if choices and len(choices) > 0:
            # Get the message content from the first choice
            message = choices[0].get('message', {})
            if message and 'content' in message:
                signature_text = message['content']
            else:
                # Try text field for completions API
                signature_text = choices[0].get('text', '')
                
    if not signature_text:
        print("SIGNATURE GENERATION FAILED - Could not extract signature text")
        print(f"Response structure: {signature_responses.keys() if isinstance(signature_responses, dict) else type(signature_responses)}")
        return False
    
    print("\nSIGNATURE OUTPUT:")
    print("-" * 50)
    print(signature_text)
    print("-" * 50)
    
    # Step 2: Run Translation
    print("\nSTEP 2: TRANSLATION")
    print("=" * 50)
    
    translation_prompt = generate_translation_prompt(test_sample, signature_text)
    
    # Check cache for translation
    translation_cache_file = get_cache_filename(args, "translation", translation_prompt, args.start_idx + example_idx)
    translation_responses = load_from_cache(translation_cache_file)
    
    if translation_responses:
        print(f"Using cached translation from {translation_cache_file}")
    else:
        print("Generating translation...")
        translation_responses = gpt_safe_completion(
            [translation_prompt], 
            temperature=args.temperature, 
            max_tokens=2048, 
            stop_token=None,
            model=args.model
        )
        # Save to cache
        save_to_cache(translation_cache_file, translation_responses)
    
    # Extract the translation text from the API response
    if not translation_responses:
        print("TRANSLATION FAILED - No response received")
        return False
    
    # Extract translation text
    translation_text = ""
    if isinstance(translation_responses, dict) and 'choices' in translation_responses:
        # Handle API response format for chat models
        print("Extracting translation from chat completion response...")
        choices = translation_responses.get('choices', [])
        if choices and len(choices) > 0:
            # Get the message content from the first choice
            message = choices[0].get('message', {})
            if message and 'content' in message:
                translation_text = message['content']
            else:
                # Try text field for completions API
                translation_text = choices[0].get('text', '')
                
    if not translation_text:
        print("TRANSLATION FAILED - Could not extract translation text")
        print(f"Response structure: {translation_responses.keys() if isinstance(translation_responses, dict) else type(translation_responses)}")
        return False
    
    print("\nTRANSLATION OUTPUT:")
    print("-" * 50)
    print(translation_text)
    print("-" * 50)
    
    # Step 3: Evaluation
    print("\nSTEP 3: EVALUATION")
    print("=" * 50)
    
    # Extract Z3 code
    code = extract_z3_code(translation_text)
    if not code:
        print("FAILED TO EXTRACT Z3 CODE")
        return False
    
    print("\nEXTRACTED Z3 CODE:")
    print("-" * 50)
    print(code)
    print("-" * 50)
    
    # Create tmp directory if it doesn't exist
    os.makedirs("tmp", exist_ok=True)
    
    # Execute Z3 code without modification
    execution_success, output = execute_z3_test(code, timeout=15.0)
    
    if not execution_success:
        print(f"EXECUTION FAILED: {output}")
        return False
    
    print("\nZ3 EXECUTION OUTPUT:")
    print("-" * 50)
    print(output)
    print("-" * 50)
    
    # Parse output to get answer
    answer = parse_z3_output(output)
    
    if not answer:
        print("COULD NOT DETERMINE ANSWER")
        return False
    
    # Check correctness
    expected_idx = test_sample.get('label')
    if expected_idx is not None:
        answer_idx = ord(answer) - ord('A')
        is_correct = (answer_idx == expected_idx)
        
        # Get the available choices
        choices = test_sample.get('choices', [])
        
        print("\nEVALUATION RESULTS:")
        print("-" * 50)
        print(f"RAW Z3 OUTPUT: {output.strip()}")
        print(f"PARSED ANSWER: {answer}")
        
        # Print predicted choice
        if 0 <= answer_idx < len(choices):
            print(f"PREDICTED CHOICE: {answer}. {choices[answer_idx]}")
        
        # Print expected choice
        print(f"EXPECTED ANSWER: {chr(ord('A') + expected_idx)}")
        if 0 <= expected_idx < len(choices):
            print(f"EXPECTED CHOICE: {chr(ord('A') + expected_idx)}. {choices[expected_idx]}")
        
        print(f"RESULT: {'CORRECT ✓' if is_correct else 'INCORRECT ✗'}")
        print("-" * 50)
        
        return {"result": "correct" if is_correct else "incorrect", "answer": answer}
    else:
        print("\nEVALUATION RESULTS:")
        print("-" * 50)
        print(f"RAW Z3 OUTPUT: {output.strip()}")
        print(f"PARSED ANSWER: {answer}")
        # Get the available choices
        choices = test_sample.get('choices', [])
        answer_idx = ord(answer) - ord('A')
        if 0 <= answer_idx < len(choices):
            print(f"PREDICTED CHOICE: {answer}. {choices[answer_idx]}")
        print("No ground truth label available")
        print("-" * 50)
        return {"result": "unknown", "answer": answer}

def main():
    """Main function to run the integrated pipeline"""
    global args
    args = setup_args()
    
    # Create cache directory if it doesn't exist
    if args.use_cache:
        os.makedirs(args.cache_dir, exist_ok=True)
    
    # Load test data
    test_data = load_test_data(args.task, args.split, args.start_idx, args.num_samples)
    
    # Create tmp directory for Z3 execution
    os.makedirs("tmp", exist_ok=True)
    
    # Initialize statistics
    stats = {
        "total": len(test_data),
        "completed": 0,
        "failed": 0,
        "correct": 0,
        "incorrect": 0,
        "unknown": 0
    }
    
    # Process each example one at a time
    for i, test_sample in enumerate(test_data):
        result = process_single_example(args, test_sample, i)
        
        if result:
            stats["completed"] += 1
            
            # Update stats based on result
            if result["result"] == "correct":
                stats["correct"] += 1
            elif result["result"] == "incorrect":
                stats["incorrect"] += 1
            else:
                stats["unknown"] += 1
        else:
            stats["failed"] += 1
    
    # Print final statistics
    print("\n" + "=" * 80)
    print("FINAL STATISTICS")
    print("=" * 80)
    print(f"Total examples: {stats['total']}")
    print(f"Completed: {stats['completed']} ({stats['completed']/stats['total']*100:.1f}%)")
    print(f"Failed: {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)")
    print(f"Correct: {stats['correct']} ({stats['correct']/stats['total']*100:.1f}%)")
    print(f"Incorrect: {stats['incorrect']} ({stats['incorrect']/stats['total']*100:.1f}%)")
    print(f"Unknown: {stats['unknown']} ({stats['unknown']/stats['total']*100:.1f}%)")
    if stats["correct"] + stats["incorrect"] > 0:
        accuracy = stats["correct"] / (stats["correct"] + stats["incorrect"]) * 100
        print(f"Accuracy: {accuracy:.1f}%")

if __name__ == "__main__":
    main() 