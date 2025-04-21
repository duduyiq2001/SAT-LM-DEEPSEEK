#!/usr/bin/env python3
"""
Simplified pipeline for SAT-LM:
1. Signature generation
2. Translation
3. Evaluation

This version processes one problem at a time and prints detailed outputs
with a proper caching mechanism.
"""

import os
import sys
import json
import logging
import argparse
import tempfile
import hashlib
from pathlib import Path
import re

# Add parent directory to path
sys.path.append('.')

# Import the evaluator functions
from evaluate_translation import extract_z3_code, execute_z3_test, parse_z3_output
from api_utils import gpt_safe_completion
from prog_solver.arlsat_parser import LSATSatProblem
from prog_solver.arlsat_solver import arlsat_satlm_exec

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

def save_to_cache(cache_file, code_text, prompt=None):
    """Save results to cache
    
    Args:
        cache_file: Path to the cache file
        response: API response to cache
        prompt: The prompt that generated this response
    """
    

        
    # Create a simplified cache entry with just code and prompt
    simplified_response = {
        'text': code_text,
        'prompt': prompt
    }
            
    with open(cache_file, 'w') as f:
            json.dump(simplified_response, f)

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

def read_manual_prompt(task, stage, prompt_id, style_template):
    """Read manual prompt from file
    
    Args:
        task (str): Task name
        stage (str): Processing stage (SIG/TRANS)
        prompt_id (str): Prompt identifier
        style_template (str): Style template
        
    Returns:
        str: Manual prompt text
    """
    prompt_file = f"manual_prompts/multistage_{task}.jsonline"
    if not os.path.exists(prompt_file):
        logger.error(f"Prompt file not found: {prompt_file}")
        sys.exit(1)
    
    with open(prompt_file, 'r') as f:
        prompt_lines = [json.loads(line) for line in f.readlines()]
    
    d = dict([(x["id"], x) for x in prompt_lines])
    if prompt_id not in d:
        logger.error(f"Prompt ID '{prompt_id}' not found in {prompt_file}")
        sys.exit(1)
    
    selected = d[prompt_id]
    assert selected["style_template"] == style_template
    return selected["prompt"]

def generate_signature_prompt(test_sample):
    """Generate a signature prompt for the given test sample"""
    # Format the test sample
    choice_str = "\n".join([f"({chr(65+i)}) {choice}" for i, choice in enumerate(test_sample.get("choices", []))])
    
    # Create a clean, focused prompt
    prompt = """# Z3 SIGNATURE GENERATION TASK
# Focus ONLY on variable declarations and type definitions

# INSTRUCTIONS:
# 1. ONLY define variables and types - NO constraints or solution logic
# 2. Setup the basic structure needed to represent the problem
# 3. DO NOT include constraint equations or validity checks

# Your signature should ONLY include:
# - Variable declarations (Bool, Int, Enum types)
# - Type definitions
# - Function declarations (if needed)

# Example signature (reference only):
### problem statement
Nine different treatments are available for a certain illness: three antibiotics—F, G, and H—three dietary regimens—M, N, and O—and three physical therapies—U, V, and W. For each case of the illness, a doctor will prescribe exactly five of the treatments, in accordance with the following conditions: If two of the antibiotics are prescribed, the remaining antibiotic cannot be prescribed. There must be exactly one dietary regimen prescribed. If O is not prescribed, F cannot be prescribed. If W is prescribed, F cannot be prescribed. G cannot be prescribed if both N and U are prescribed. V cannot be prescribed unless both H and M are prescribed.
Question: If O is prescribed for a given case, which one of the following is a pair of treatments both of which must also be prescribed for that case?
Choices:
(A) F, M
(B) G, V
(C) N, U
(D) U, V
(E) U, W
### signature
# declare variables
treatments = EnumSort([F, G, H, M, N, O, U, V, W])
antibiotics = EnumSort([F, G, H])
dietary_regimens = EnumSort([M, N, O])
physical_therapies = EnumSort([U, V, W])
prescribed = Function(treatments, bool)

# Question: If O is prescribed for a given case, which one of the following is a pair of treatments both of which must also be prescribed for that case?
# we check whether the options must be true
print(check_valid())
# REMEMBER:
# - ONLY provide variable and type definitions
# - DO NOT include constraints or solution logic
# - DO NOT attempt to solve the problem

# Problem to model:
"""
    
    # Add the test sample to the prompt
    prompt += f"\"\"\"\n{test_sample['context']}\nQuestion: {test_sample['question']}\nChoices:\n{choice_str}\n\"\"\"\n"
    
    return prompt

def generate_translation_prompt(test_sample, signature_code):
    """Generate a translation prompt for the given test sample and signature"""
    # Format the test sample
    choice_str = "\n".join([f"({chr(65+i)}) {choice}" for i, choice in enumerate(test_sample.get("choices", []))])
    
    problem_text = f"{test_sample['context']}\nQuestion: {test_sample['question']}\nChoices:\n{choice_str}"
    
    # Extract just the variable declarations from the signature
    lines = signature_code.split("\n")
    signature_lines = []
    variable_section = False
    
    for line in lines:
        if "Students =" in line or "Days =" in line or "Times =" in line or "schedule =" in line or "gives_report =" in line:
            signature_lines.append(line.strip())
            variable_section = True
        elif variable_section and line.strip() and not line.strip().startswith('#') and "check_valid" not in line and "print" not in line:
            signature_lines.append(line.strip())
    
    signature_part = "\n".join(signature_lines)
    if not signature_part.strip():
        signature_part = "# Failed to extract variable declarations from signature"
    
    # Create translation prompt with clear code block markers
    prompt = f"""# Z3 TRANSLATION TASK
# Follow these instructions EXACTLY:
# 1. Implement a solution using the variables in the signature below
# 2. Place your complete solution between ```python and ``` markers ONLY
# 3. Imports MUST be at module level, not inside functions
# 4. Return ONLY the option letter (A-E) that satisfies the constraints
# 5. make sure your solution is using correct python syntax and is executable code
# 6. for detailed documentation for z3, refer to https://ericpony.github.io/z3py-tutorial/guide-examples.html

PROBLEM:
\"\"\"
{problem_text}
\"\"\"

VARIABLE DEFINITIONS FROM SIGNATURE:
\"\"\"
{signature_part}
\"\"\"

INSTRUCTIONS:
- Implement constraints based on problem description
- Check each option systematically
- Return only the letter of the valid option
- IMPORTANT: Place code between ```python and ``` markers

EXAMPLE FORMAT:
```python
from z3 import *

# Variable definitions from signature
{signature_part}

def check_valid():
    # Your implementation here
    return "A"  # Return only the option letter

if __name__ == "__main__":
    result = check_valid()
    print(result)
```

NOW IMPLEMENT YOUR SOLUTION:
"""
    return prompt

def extract_clean_code(response_text):
    """
    Enhanced extraction function that isolates Python code blocks
    marked with ```python and ``` from the response text.
    
    Args:
        response_text (str): The full response text from the model
        
    Returns:
        str: The extracted code, or None if no valid code block found
    """
    # Match the code block after "NOW IMPLEMENT YOUR SOLUTION:"
    import re
    pattern = r"NOW IMPLEMENT YOUR SOLUTION:\s*```python\s*([\s\S]*?)\s*```"
    matches = re.findall(pattern, response_text)
    
    if matches and len(matches) > 0:
        code = matches[0]
        return code
    
    # Fall back to the original extract_z3_code logic if no python code blocks found
    return extract_z3_code(response_text)

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
    cached_data = load_from_cache(signature_cache_file)
    
    if cached_data:
        print(f"Using cached signature from {signature_cache_file}")
        signature_text = cached_data.get('text', '')
    else:
        print("Generating signature...")
        signature_responses = gpt_safe_completion(
            [signature_prompt], 
            temperature=args.temperature, 
            max_tokens=2048, 
            stop_token=None,
            model=args.model
        )
        # # Save to cache with the prompt
        # save_to_cache(signature_cache_file, signature_responses, signature_prompt)
        
        # Extract the signature text from the API response
        if not signature_responses:
            print("SIGNATURE GENERATION FAILED - No response received")
            return False
        
        # Extract text from response
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
    # We'll use the LSATSatProblem parser to validate the signature
    try:
        signature_code = extract_z3_code(signature_text)
        if signature_code:
            print(signature_code)
            save_to_cache(signature_cache_file, signature_code, signature_prompt)
        else:
            print(signature_text)
            save_to_cache(signature_cache_file, signature_text, signature_prompt)
            print("WARNING: Could not extract clean Z3 code from signature")
    except Exception as e:
        print(signature_text)
        print(f"WARNING: Error parsing signature: {str(e)}")
    print("-" * 50)
    
    # Step 2: Run Translation
    print("\nSTEP 2: TRANSLATION")
    print("=" * 50)
    
    translation_prompt = generate_translation_prompt(test_sample, signature_text)
    
    # Check cache for translation
    translation_cache_file = get_cache_filename(args, "translation", translation_prompt, args.start_idx + example_idx)
    cached_data = load_from_cache(translation_cache_file)
    
    if cached_data:
        print(f"Using cached translation from {translation_cache_file}")
        translation_text = cached_data.get('text', '')
    else:
        print("Generating translation...")
        translation_responses = gpt_safe_completion(
            [translation_prompt], 
            temperature=args.temperature, 
            max_tokens=2048, 
            stop_token=None,
            model=args.model
        )
        # # Save to cache with the prompt
        # save_to_cache(translation_cache_file, translation_responses, translation_prompt)
        
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
    
    
    # Step 3: Evaluation using arlsat_satlm_exec
    print("\nSTEP 3: EVALUATION")
    print("=" * 50)
    
    # First try to extract a clean code block using our enhanced extractor
    clean_code = extract_clean_code(translation_text)
    if clean_code:
        print("Extracted clean code block from response")
        print(clean_code)
        save_to_cache(translation_cache_file, clean_code, translation_prompt)
        success, output = execute_z3_test(clean_code)
    else:
        print("WARNING: Could not extract clean code block, trying full response")
        # Fall back to original behavior
        success, output = execute_z3_test(translation_text)
    
    if not success:
        print(f"EXECUTION FAILED: {output}")
        
        # Fallback to standard execution
        print("\nTrying fallback execution...")
        if clean_code:
            code = clean_code
        else:
            code = extract_z3_code(translation_text)
            
        if not code:
            print("FAILED TO EXTRACT Z3 CODE")
            return False
        
        # Create tmp directory if it doesn't exist
        os.makedirs("tmp", exist_ok=True)
        
        execution_success, output = execute_z3_test(code, timeout=15.0)
        if not execution_success:
            print(f"FALLBACK EXECUTION FAILED: {output}")
            return False
    
    print("\nZ3 EXECUTION OUTPUT:")
    print("-" * 50)
    if isinstance(output, list):
        print("\n".join(output))
    else:
        print(output)
    print("-" * 50)
    
    # Parse output to get answer
    if isinstance(output, list):
        # If output is a list of lines, find the first line with a letter
        for line in output:
            if line.strip() in ["A", "B", "C", "D", "E"]:
                answer = line.strip()
                break
        else:
            # If no direct letter, try to parse
            answer = parse_z3_output("\n".join(output))
    else:
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
        if isinstance(output, list):
            print(f"RAW Z3 OUTPUT: {output}")
        else:
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
        if isinstance(output, list):
            print(f"RAW Z3 OUTPUT: {output}")
        else:
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