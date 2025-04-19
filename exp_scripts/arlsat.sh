#!/bin/bash
# ArLSAT Evaluation Script
# 
# This script runs both the manual prompting approach (CoT) and the multi-stage approach 
# on the ArLSAT dataset to compare their performance.
#
# Usage: 
#   bash exp_scripts/arlsat.sh [eval_split] [num_dev]
#
# Arguments:
#   eval_split: Evaluation split to use (default: "test")
#   num_dev: Number of development examples to use (default: -1, which means all)
#
# Environment variables:
#   KEY: OpenAI API key
#   FLAG: Additional flags for the commands (optional)

# Set evaluation split (default: "test")
EVAL=${1:-"test"}

# Set number of development examples (default: "-1", meaning all examples)
NUM_DEV=${ND:-"-1"}

# Set the model to use
MODEL="gpt-4.1"

# Set resume flag (default: empty, meaning don't resume)
RESUME=${RESUME:-""}

# Check if python-dotenv is installed, and install it if it's not
pip install python-dotenv 2>/dev/null || true

# Ensure the OpenAI API key is available
if [ -f ".env" ]; then
    echo "Using API key from .env file"
    export $(grep -v '^#' .env | xargs)
elif [ -n "$KEY" ]; then
    echo "Using API key from KEY environment variable"
    export OPENAI_API_KEY=$KEY
else
    echo "Error: No OpenAI API key found. Please set it in your environment variables or .env file."
    exit 1
fi

# Test the API key
python3 -c "
from openai import OpenAI
import os
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
try:
    models = client.models.list()
    print('API key is valid!')
except Exception as e:
    print(f'Error testing API key: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "Error: Could not connect to OpenAI API. Please check your API key."
    exit 1
fi

main_exp()
{
    # Run Chain-of-Thought (CoT) approach using manual prompting
    # Parameters:
    #   --task arlsat: Specifies the ArLSAT task
    #   --run_prediction: Run prediction (not just evaluation)
    #   --batch_size 5: Process 5 examples at a time
    #   --num_samples 1: Generate 1 sample per example
    #   --temperature 0.0: Use deterministic sampling
    #   --style_template cot: Use Chain-of-Thought template
    #   --manual_prompt_id cot: Use CoT prompt
    #   --model: Use the specified model
    #OPENAI_API_KEY=${OPENAI_API_KEY} python3 run_manual.py --task arlsat --run_prediction --batch_size 5 --num_samples 1 --temperature 0.0 --style_template cot --manual_prompt_id cot --num_dev ${NUM_DEV} --do_impose_prediction ${FLAG} --model ${MODEL} --force_override
    
    # Run Multi-stage approach (signature generation + translation)
    # Parameters:
    #   --task arlsat: Specifies the ArLSAT task
    #   --run_prediction: Run prediction (not just evaluation)
    #   --batch_size 5: Process 5 examples at a time
    #   --num_samples 1: Generate 1 sample per example
    #   --temperature 0.0: Use deterministic sampling
    #   --sig_prompt_id sigz3: Use Z3-based signature prompt
    #   --trans_setting setupsatlm: Use SATLM translation setting
    #   --model: Use the specified model
    RESUME_FLAG=""
    FORCE_OVERRIDE_FLAG=""
    
    if [ -n "$RESUME" ]; then
        RESUME_FLAG="--resume"
        echo "Resuming from previous run"
    else
        FORCE_OVERRIDE_FLAG="--force_override"
    fi
    
    OPENAI_API_KEY=${OPENAI_API_KEY} python3 run_multistage.py --task arlsat --run_prediction --batch_size 5 --num_samples 1 --temperature 0.0 --sig_prompt_id sigz3 --trans_setting setupsatlm --eval_split ${EVAL} --num_dev ${NUM_DEV} --do_impose_prediction ${FLAG} --model ${MODEL} ${FORCE_OVERRIDE_FLAG} ${RESUME_FLAG}
}

# Execute the main experiment function
main_exp
