"""
API utilities for handling OpenAI API requests, caching, and token management.
This module provides utilities for interacting with OpenAI's API, including
request formatting, caching, error handling, and token counting.
"""
import os
import itertools
import time
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
import random
import tiktoken

from tqdm import tqdm
from utils import *

# Load environment variables from .env file
load_dotenv()

# Error identifier for OpenAI API errors
API_ERROR_IDENTIFIER = "OPENAI Error"
# Load tokenizer for token counting
_TOKENIZER = tiktoken.encoding_for_model("gpt-4")
# Constants for API requests
GPT3_LENGTH_LIMIT = 2049
GPT_MAX_ATTEMPTS = 60
GPT_WAITTIME = 20
API_ERROR_IDENTIFIER = "OPENAI Error"

# Initialize OpenAI client
_client = None

def register_query_args(parser):
    """
    Register command-line arguments related to API queries.
    
    Args:
        parser (argparse.ArgumentParser): The argument parser to extend
    """
    # parser.add_argument('--engine', default='code-davinci-002', choices=[
    #     "txext-davinci-002", "text-davinci-003", "code-davinci-001", "code-davinci-002"])
    parser.add_argument('--run_prediction', default=False, action='store_true')
    parser.add_argument('--do_dryrun', default=False, action='store_true')
    parser.add_argument('--force_override', default=False, action='store_true')
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--model', type=str, default="gpt-4-turbo")

def register_base_args(parser):
    """
    Register base command-line arguments for all tasks.
    
    Args:
        parser (argparse.ArgumentParser): The argument parser to extend
    """
    # standard, instruction, etc
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--eval_split', type=str, default="test")
    parser.add_argument('--slice_train', type=int, default=0)
    parser.add_argument('--num_train', type=int, default=-1)
    parser.add_argument('--num_shots', type=int, default=5)
    parser.add_argument('--slice_dev', type=int, default=0)
    parser.add_argument('--num_dev', type=int, default=-1)
    parser.add_argument('--do_print', default=False, action='store_true')
    parser.add_argument('--num_eval_samples', type=int, default=-1)
    parser.add_argument('--first_k', type=int, default=-1)
    parser.add_argument('--do_impose_prediction', default=False, action='store_true')
    register_query_args(parser)


def config_args_and_api(args):
    """
    Configure arguments and set up the OpenAI API.
    
    Args:
        args: Command line arguments
    """
    global _client
    if args.batch_size == -1:
        args.batch_size = 1
    
    # Try to get API key from multiple sources
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("KEY")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in your environment or .env file.")
    
    _client = OpenAI(api_key=api_key)

def gpt_style_tokenize(x):
    """
    Tokenize text using the tiktoken tokenizer.
    
    Args:
        x (str): The input text to tokenize
        
    Returns:
        list: The tokenized text
    """
    return _TOKENIZER.encode(x)

def length_of_prompt(prompt, max_tokens):
    """
    Calculate the total length (in tokens) of a prompt plus expected completion.
    
    Args:
        prompt (str): The prompt text
        max_tokens (int): The maximum number of tokens for completion
        
    Returns:
        int: Total token count (prompt + max_tokens)
    """
    return len(_TOKENIZER.encode(prompt)) + max_tokens

def generate_fake_logprobs(text, prompt_len=0):
    """
    Generate fake logprobs for models that don't support them.
    
    Args:
        text (str): Text to generate logprobs for
        prompt_len (int): Length of the prompt
        
    Returns:
        dict: Fake logprobs structure
    """
    tokens = _TOKENIZER.encode(text)
    
    # Create reasonable text offsets that increase with token position
    # This is an approximation as we don't have actual character offsets
    text_len = len(text)
    token_len = len(tokens)
    if token_len > 0:
        # Create approximately evenly spaced offsets
        token_offset = [int(i * text_len / token_len) for i in range(token_len)]
    else:
        token_offset = []
        
    token_logprobs = [random.uniform(-2.0, -0.1) for _ in range(len(tokens))]
    
    # Decode tokens to strings for compatibility with existing code
    decoded_tokens = [_TOKENIZER.decode([token]) for token in tokens]
    
    return {
        "tokens": decoded_tokens,
        "token_logprobs": token_logprobs,
        "text_offset": token_offset,
        "top_logprobs": None
    }

def gpt_safe_completion(prompts, temperature, max_tokens, stop_token, logprobs=1, num_samples=1, echo=True, model="gpt-4-turbo"):
    """
    Make a safe API call to OpenAI's completion endpoint with error handling.
    
    Args:
        prompts (list): List of prompt strings
        temperature (float): Sampling temperature
        max_tokens (int): Maximum tokens to generate
        stop_token (str): Token to stop generation
        logprobs (int): Number of log probabilities to return
        num_samples (int): Number of completions per prompt
        echo (bool): Whether to echo prompt in response
        model (str): Model to use for completion
        
    Returns:
        dict: OpenAI API response or error response
    """
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    last_exc = None
    
    # Check if model is a chat model
    is_chat_model = any(x in model for x in ["gpt-4", "gpt-3.5-turbo", "gpt-4.1"]) and "instruct" not in model
    
    print(f"Using model: {model} (is_chat_model: {is_chat_model})")
    
    # Ensure max_tokens is appropriate for the model
    # Different models have different context limits
    model_max_tokens = {
        "gpt-4-turbo": 4096,
        "gpt-4": 2048,
        "gpt-4.1": 8192,  # Increased limit for GPT-4.1
        "gpt-3.5-turbo": 2048
    }
    
    # Get default max tokens if model not in dictionary
    default_max = 2048
    model_limit = model_max_tokens.get(model, default_max)
    
    # Ensure max_tokens doesn't exceed model limit
    if max_tokens > model_limit:
        print(f"Warning: Requested max_tokens ({max_tokens}) exceeds model limit. Capping to {model_limit}")
        max_tokens = model_limit
    
    # Create a new format for choices that matches the old API's format
    all_choices = []
    
    for attempt in range(GPT_MAX_ATTEMPTS):
        try:
            all_choices = []
            
            for prompt in prompts:
                prompt_choices = []
                
                for _ in range(num_samples):
                    if is_chat_model:
                        # Use chat completions API for chat models
                        response = _client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=temperature,
                            max_tokens=max_tokens,
                            stop=stop_token
                        )
                        
                        completion_text = response.choices[0].message.content
                        
                        # Create a backwards-compatible format
                        choice = {
                            "text": completion_text,
                            "prompt": prompt,
                            "completion_offset": len(prompt) if echo else 0
                        }
                        
                        # If echo is True, prepend the prompt to the text
                        if echo:
                            choice["text"] = prompt + choice["text"]
                        
                        # Generate fake logprobs for chat models
                        choice["logprobs"] = generate_fake_logprobs(choice["text"], len(prompt))
                            
                    else:
                        # Use completions API for non-chat models
                        response = _client.completions.create(
                            model=model,
                            prompt=prompt,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            logprobs=logprobs,
                            echo=echo,
                            stop=stop_token
                        )
                        
                        # Create a backwards-compatible format
                        choice = {
                            "text": response.choices[0].text,
                            "prompt": prompt,
                            "logprobs": response.choices[0].logprobs,
                            "completion_offset": len(prompt) if echo else 0
                        }
                    
                    prompt_choices.append(choice)
                
                all_choices.extend(prompt_choices)
            
            return {"choices": all_choices}
        
        except Exception as e:
            print(f"OpenAI API error (attempt {attempt+1}/{GPT_MAX_ATTEMPTS}): {str(e)}")
            last_exc = e
            time.sleep(GPT_WAITTIME)
    
    # Make a fake response after all attempts failed
    fake_choices = [
        {
            "text": p + " OPENAI Error - " + str(last_exc),
            "prompt": p,
            "logprobs": generate_fake_logprobs(p + " OPENAI Error - " + str(last_exc)),
            "API Error": True,
        } for p in prompts for _ in range(num_samples)
    ]
    
    return {"choices": fake_choices}

def batch_query_engine(args, prompts, max_tokens, stop_token):
    """
    Process multiple prompts in batches and manage API responses.
    
    Args:
        args: Command line arguments
        prompts (list): List of prompt strings
        max_tokens (int): Maximum tokens to generate
        stop_token (str): Token to stop generation
        
    Returns:
        list: Processed API responses
    """
    model = args.model if hasattr(args, "model") else "gpt-4-turbo"
    
    # Determine which stage we're in based on the cache filename
    current_stage = "UNKNOWN"
    current_examples = "UNKNOWN"
    if hasattr(args, "sig_prompt_id") and hasattr(args, "sig_method"):
        if "multisgate-sig-" in args._cache_filename:
            current_stage = "SIGNATURE"
        elif "multisgate-trans-" in args._cache_filename:
            current_stage = "TRANSLATION"
            
    print(f"\n{current_stage} STAGE: Querying model {model} for {len(prompts)} prompts")
    print(f"{current_stage} STAGE: Max tokens: {max_tokens}")
    
    resps = gpt_safe_completion(
        prompts=prompts, 
        temperature=args.temperature, 
        max_tokens=max_tokens, 
        stop_token=stop_token, 
        logprobs=1, 
        num_samples=args.num_samples, 
        echo=True,
        model=model
    )

    resps = resps["choices"]
    # Hide raw API response debug info to avoid terminal spam
    # print("RESPS", resps)
    # print("P", prompts, len(prompts))
    
    # Convert to list if it's not already and group by prompt
    resps = [resps[(i * args.num_samples):(i * args.num_samples + args.num_samples)] for i in range(len(prompts))]
    # print(resps, len(resps))
    
    for i, (prompt, resp) in enumerate(zip(prompts, resps)):
        print(f"{current_stage} STAGE: Processing response {i+1}/{len(prompts)}")
        for pred in resp:
            pred["prompt"] = prompt
            if len(pred["text"]) > len(prompt):
                pred["text"] = pred["text"][len(prompt):]
            else:
                pred["text"] = " NULL"
            pred["completion_offset"] = len(prompt)
            
            # Print just the completion text to see responses clearly
            if "API Error" not in pred:
                print(f"----- {current_stage} Model Response for Prompt {i+1}/{len(prompts)} -----")
                print(pred["text"][:200] + "..." if len(pred["text"]) > 200 else pred["text"])
                print("--------------------------")

    return resps

# args
# prompts: 2d array
# cache_filename
# for gpt, assuming apis are pretty robust
def run_completion_tasks_with_cache(args, cache_fileneme, prompts_by_examples, max_tokens, stop_token):
    """
    Run completion tasks with caching to avoid redundant API calls.
    Writes results to cache after each batch for immediate viewing.
    
    Args:
        args: Command line arguments
        cache_fileneme (str): File to cache results
        prompts_by_examples (list): 2D array of prompts by examples
        max_tokens (int): Maximum tokens to generate
        stop_token (str): Token to stop generation
        
    Returns:
        list: Processed API responses (from cache or new requests)
    """
    # Store the cache filename in args for use in other functions
    args._cache_filename = cache_fileneme
    
    # Determine stage from cache filename for logging
    stage = "UNKNOWN"
    if "multisgate-sig-" in cache_fileneme:
        stage = "SIGNATURE"
    elif "multisgate-trans-" in cache_fileneme:
        stage = "TRANSLATION"
    elif "manual" in cache_fileneme:
        stage = "MANUAL"
    
    print(f"\n{'='*50}")
    print(f"{stage} STAGE: STARTING CACHE/API OPERATIONS")
    print(f"{stage} STAGE: Cache file: {cache_fileneme}")
    print(f"{stage} STAGE: args.force_override = {getattr(args, 'force_override', 'NOT SET')}")
    print(f"{stage} STAGE: args.resume = {getattr(args, 'resume', 'NOT SET')}")
    print(f"{'='*50}\n")
    
    # Make the assertion more robust to handle empty lists
    assert isinstance(prompts_by_examples, list), "prompts_by_examples must be a list"
    if prompts_by_examples:
        assert isinstance(prompts_by_examples[0], list), "prompts_by_examples[0] must be a list"
        if prompts_by_examples[0]:
            assert isinstance(prompts_by_examples[0][0], str), "prompts_by_examples[0][0] must be a string"
    
    if max_tokens == 0:
        assert args.num_samples == 1
    shape_records = [len(x) for x in prompts_by_examples]
    data_size = sum(shape_records)

    # Check if we have a complete cache
    if os.path.exists(cache_fileneme):
        print(f"{stage} STAGE: Cached Predictions Detected: {cache_fileneme}")
        if args.force_override:
            print(f"{stage} STAGE: Force Overriding Previous Predictions")
        else:
            print(f"{stage} STAGE: 📂 Loading complete cache file, skipping API calls")
            return read_json(cache_fileneme)
    
    # Check if we're resuming and have partial results
    partial_cache_filename = cache_fileneme + ".partial"
    incremental_cache = [[] for _ in range(len(prompts_by_examples))]
    example_counts = [0] * len(prompts_by_examples)
    
    if hasattr(args, 'resume') and args.resume and os.path.exists(partial_cache_filename):
        try:
            print(f"{stage} STAGE: Loading partial results from {partial_cache_filename}")
            partial_results = read_json(partial_cache_filename)
            if isinstance(partial_results, list) and len(partial_results) == len(prompts_by_examples):
                # Copy over the existing results
                for i, results in enumerate(partial_results):
                    if results:  # Skip empty placeholder results
                        incremental_cache[i] = results
                        example_counts[i] = len(results)
                print(f"{stage} STAGE: Loaded partial results for {sum(1 for c in example_counts if c > 0)} examples")
        except Exception as e:
            print(f"{stage} STAGE: Error loading partial results: {e}")
            # Continue with empty incremental_cache

    samples = list(itertools.chain(*prompts_by_examples))

    renewed_results = []
    prompt_lengths = []
    request_pool = []

    task_max_tokens = max_tokens
    for idx, prompt in enumerate(samples):
        # Skip empty prompts (placeholders for already processed examples)
        if not prompt:
            continue
            
        if args.do_dryrun:
            response = length_of_prompt(prompt, task_max_tokens)
            print("-----------------------------------------")
            print(prompt)
            print("LEN", response)
            prompt_lengths.append(response)

        # add to request pool if no cached results, or error happened
        request_pool.append((idx, prompt))

    if args.do_dryrun:
        print(cache_fileneme)
        print('Total request', len(request_pool))
        print('MAX', max(prompt_lengths), 'COMP', task_max_tokens)
        return

    num_request, batch_size = len(request_pool), args.batch_size
    num_batch = (num_request + batch_size - 1) // batch_size
    
    # Build a mapping from global index to (example_idx, local_idx)
    global_idx_to_example_idx = {}
    global_idx = 0
    for ex_idx, prompts in enumerate(prompts_by_examples):
        for local_idx, _ in enumerate(prompts):
            global_idx_to_example_idx[global_idx] = (ex_idx, local_idx)
            global_idx += 1
    
    # prediction loop, auto managing batching for OPT
    print(f"{stage} STAGE: Processing {num_request} total requests in batches of {batch_size}")
    for batch_idx in tqdm(range(num_batch), total=num_batch, desc=f"{stage} Querying"):
        batch_start = batch_idx * batch_size
        batch_end = batch_start + batch_size
        reqs = request_pool[batch_start: batch_end]

        idx_lists = [x[0] for x in reqs]
        prompts = [x[1] for x in reqs]

        responses = batch_query_engine(args, prompts, task_max_tokens, stop_token)

        assert len(idx_lists) == len(responses)
        
        for i, resp in zip(idx_lists, responses):
            renewed_results.append(resp)
            
            # Get example_idx and local_idx
            if i in global_idx_to_example_idx:
                ex_idx, local_idx = global_idx_to_example_idx[i]
                
                # If incremental_cache[ex_idx] is shorter than we need, extend it
                while len(incremental_cache[ex_idx]) <= local_idx:
                    incremental_cache[ex_idx].append([])
                    
                # Add the response
                incremental_cache[ex_idx][local_idx] = resp
                example_counts[ex_idx] += 1
                
                # Write the current state to cache after each response (not just completed examples)
                partial_cache = []
                for j, results in enumerate(incremental_cache):
                    partial_cache.append(results)  # Include all results, even incomplete
                
                print(f"{stage} STAGE: Writing partial results after response {i} ({sum(example_counts)}/{data_size} responses complete)")
                dump_json(partial_cache, cache_fileneme + ".partial")
                print(f"{stage} STAGE: 📝 Cached partial results to {cache_fileneme}.partial")
                
                # Also create a human-readable version with just the text
                readable_results = []
                for result_list in partial_cache:
                    if result_list:
                        example_texts = []
                        for result in result_list:
                            if result:
                                example_texts.append(result[0]["text"] if result else "")
                            else:
                                example_texts.append("")
                        readable_results.append(example_texts)
                    else:
                        readable_results.append([])
                
                # Write readable version
                with open(cache_fileneme + ".readable", "w") as f:
                    for i, example in enumerate(readable_results):
                        if example:
                            f.write(f"Example {i}:\n")
                            for j, text in enumerate(example):
                                if text:
                                    f.write(f"Response {j}:\n{text}\n\n")
                            f.write("-" * 80 + "\n\n")
                print(f"{stage} STAGE: 📖 Updated readable file at {cache_fileneme}.readable")
                
                # Original code that only writes after an example is complete
                if example_counts[ex_idx] == shape_records[ex_idx]:
                    print(f"{stage} STAGE: ✅ EXAMPLE {ex_idx} COMPLETE ({sum(1 for x in example_counts if x == shape_records[x])}/{len(incremental_cache)} examples complete)")

    # Final processing 
    total_expected = sum(shape_records)
    if len(renewed_results) < total_expected and hasattr(args, 'resume') and args.resume:
        print(f"{stage} STAGE: Processed {len(renewed_results)} of {total_expected} expected results")
        print(f"{stage} STAGE: Merging with partial results from previous runs...")
        
        # Create complete results by filling gaps with what we have in incremental_cache
        complete_results = []
        
        # Flatten the incremental_cache
        for example_results in incremental_cache:
            for result in example_results:
                complete_results.append(result)
        
        if len(complete_results) > len(renewed_results):
            renewed_results = complete_results
            print(f"{stage} STAGE: Successfully merged results, now have {len(renewed_results)} results")
    
    # Final consistency check and reshaping
    if len(renewed_results) < total_expected:
        print(f"{stage} STAGE: WARNING: Missing {total_expected - len(renewed_results)} results!")
        # Fill in missing results with empty responses
        while len(renewed_results) < total_expected:
            renewed_results.append([{"text": "MISSING RESULT"}])

    # group by example
    slice_start = 0
    renewed_cache = []
    for n in shape_records:
        renewed_cache.append(renewed_results[slice_start: slice_start + n])
        slice_start = slice_start + n

    dump_json(renewed_cache, cache_fileneme)
    print(f"{stage} STAGE: 🔄 FINAL RESULTS WRITTEN to {cache_fileneme}")
    
    # Also create final human-readable version
    readable_results = []
    for result_list in renewed_cache:
        example_texts = []
        for result in result_list:
            example_texts.append(result[0]["text"] if result else "")
        readable_results.append(example_texts)
    
    # Write readable version
    with open(cache_fileneme + ".readable", "w") as f:
        for i, example in enumerate(readable_results):
            f.write(f"Example {i}:\n")
            for j, text in enumerate(example):
                f.write(f"Response {j}:\n{text}\n\n")
            f.write("-" * 80 + "\n\n")
    print(f"{stage} STAGE: 📚 FINAL READABLE RESULTS written to {cache_fileneme}.readable")
    
    print(f"\n{'='*50}")
    print(f"{stage} STAGE: CACHE/API OPERATIONS COMPLETED")
    print(f"{'='*50}\n")
    
    return renewed_cache


def score_of_completion(response):
    """
    Calculate the score (log probability sum and mean) for a completion.
    
    Args:
        response (dict): API response containing completion
        
    Returns:
        tuple: (sum of log probs, mean of log probs)
    """
    if "logprobs" not in response or response["logprobs"] is None:
        return .0, .0

    completion_offset = response.get("completion_offset", len(response["prompt"]))
    tokens = response["logprobs"]["tokens"]
    token_offset = response["logprobs"]["text_offset"]

    if completion_offset in token_offset:
        completion_start_tok_idx = token_offset.index(completion_offset)
    elif completion_offset > token_offset[-1]:
        completion_start_tok_idx = len(token_offset)
    else:
        completion_start_tok_idx = next(filter(lambda x: token_offset[x] >= completion_offset, range(len(token_offset))))

    if "<|endoftext|>" in tokens:
        completion_end_tok_idx = tokens.index("<|endoftext|>", completion_start_tok_idx)
    else:
        complention_end_offset = completion_offset + len(response["text"])
        try:
            completion_end_tok_idx = next(filter(lambda x: token_offset[x + 1] >= complention_end_offset, range(len(token_offset) - 1)), len(token_offset))
        except IndexError:
            completion_end_tok_idx = len(token_offset) - 1

    tok_scores = response["logprobs"]["token_logprobs"][completion_start_tok_idx:completion_end_tok_idx + 1]
    toks = response["logprobs"]["tokens"][completion_start_tok_idx:completion_end_tok_idx + 1]

    # Handle empty scores or None values
    if not tok_scores or None in tok_scores:
        return 0.0, 0.0
        
    tok_scores = np.array([s for s in tok_scores if s is not None])
    if len(tok_scores) == 0:
        return 0.0, 0.0
        
    return tok_scores.sum(), tok_scores.mean()

def confidence_of_completion(response, answer_hint):
    """
    Calculate a confidence score for a specific answer within a completion.
    
    Args:
        response (dict): API response containing completion
        answer_hint (str): The answer to locate in the completion
        
    Returns:
        float: Confidence score for the answer
    """
    if "logprobs" not in response or response["logprobs"] is None:
        return 0.0
        
    completion_offset = response.get("completion_offset", len(response["prompt"]))
    tokens = response["logprobs"]["tokens"]
    token_offset = response["logprobs"]["text_offset"]

    # answer_offset = response["text"]
    lower_text = response["text"].lower()
    lower_hint = answer_hint.lower()
    if lower_hint in lower_text:
        answer_offset = completion_offset + lower_text.index(lower_hint) + len(lower_hint)
    else:
        answer_offset = completion_offset

    try:
        if answer_offset in token_offset:
            answer_start_tok_idx = token_offset.index(answer_offset)
        elif answer_offset >= token_offset[-1]:
            return 0.
        else: 
            answer_start_tok_idx = next(filter(lambda x: token_offset[x] >= answer_offset, range(len(token_offset))))

        if "<|endoftext|>" in tokens:
            answer_end_tok_idx = tokens.index("<|endoftext|>", answer_start_tok_idx)
        elif "\n" in tokens[answer_start_tok_idx:]:
            answer_end_tok_idx = tokens.index("\n", answer_start_tok_idx)
        else:
            answer_end_tok_idx = len(tokens)
        if tokens[answer_end_tok_idx - 1].strip() == '.':
            answer_end_tok_idx = answer_end_tok_idx - 1

        # completion_end_tok_idx = tokens.index("<|endoftext|>")
        # return len(tokens) - completion_start_tok_idx

        tok_scores = response["logprobs"]["token_logprobs"][answer_start_tok_idx:answer_end_tok_idx]
        toks = response["logprobs"]["tokens"][answer_start_tok_idx:answer_end_tok_idx]
        
        # Handle None values and empty lists
        if not tok_scores or None in tok_scores:
            return 0.0
            
        tok_scores = np.array([s for s in tok_scores if s is not None])
        if len(tok_scores) == 0:
            return 0.0
            
        conf = np.exp(np.sum(tok_scores))
        # print("".join(toks), conf)

        return conf
    except (IndexError, ValueError):
        return 0.0
