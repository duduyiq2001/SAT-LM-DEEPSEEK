"""
Multi-stage prompting for solving logical reasoning problems.

This module implements a two-stage approach:
1. Signature Generation: Create formal representations of problems
2. Translation: Convert signatures to executable Z3 code

The workflow processes natural language problems through language models
to generate formal specifications and executable code for automated reasoning.
"""
import os
import argparse
import itertools
from random import choices

from tqdm import tqdm
from math import ceil

import numpy as np

from utils import *
from api_utils import (
    run_completion_tasks_with_cache,
    config_args_and_api,
    register_base_args,
)

from task_helper import TaskHelper, load_train_test_set
from run_manual import run_evaluation, get_eval_split_abbrev
from task_evaluator import TaskEvaluator, get_task_evaluator, Prediction, print_tabular_results

def register_multistage_args(parser):
    """Registers multi-stage specific command-line arguments
    
    Args:
        parser (argparse.ArgumentParser): Parent parser to extend
        
    Adds:
        --sig_method: Signature generation approach
        --sig_style_template: Prompt styling template
        --num_trans_shots: Number of examples for translation
        --trans_setting: Translation configuration preset
        --resume: Resume from partial results
    """
    parser.add_argument('--sig_method', type=str, default="manual", choices=["manual"])
    parser.add_argument('--sig_style_template', type=str, default="sigtpl")
    parser.add_argument('--sig_prompt_id', type=str, default="sigz3")

    parser.add_argument('--num_trans_shots', type=int, default=3)
    parser.add_argument('--trans_setting', type=str, default="setupsatlm", choices=["setupsatlm",])
    parser.add_argument('--resume', action='store_true', default=False, help='Resume from partial results')


SIG_STAGE = "SIG"
TRANA_STAGE = "TRANS"

TRANS_ANNOTATION_DIR = "annotations"

class MultiStageTaskHelper:
    """Base class for stage-specific handlers
    
    This is an abstract base class that provides the interface for
    stage-specific task helpers.
    
    Attributes:
        style_to_completion_length (dict): Maps style to completion length
        style_to_train_sep (dict): Maps style to training example separator
        style (str): The style template being used
    """

    style_to_completion_length = {}
    style_to_train_sep = {}

    def __init__(self, style):
        self.style = style

    @classmethod
    def from_taskname(cls, taskname, style):
            raise RuntimeError("Not Implemented Yet")

    def prompt_func(self, test_ex, shots):
        """Abstract method to create prompts
        
        Args:
            test_ex: Test example to create prompt for
            shots: Shot examples to include in prompt
            
        Raises:
            RuntimeError: When not implemented in subclass
        """
        raise RuntimeError("Not Implemented Yet")

    def get_completion_length(self):
        """Get expected completion length for current style
        
        Returns:
            int: Maximum token length for completion
        """
        return self.style_to_completion_length[self.style]

    def get_train_sep(self):
        """Get separator for training examples for current style
        
        Returns:
            str: Separator string
        """
        return self.style_to_train_sep[self.style]


class SigStageHelper(MultiStageTaskHelper):
    """Helper for the signature generation stage
    
    This class specializes MultiStageTaskHelper for the signature
    generation stage of the multi-stage process.
    """
    
    @classmethod
    def from_taskname(cls, taskname, style):
        """Factory method to create appropriate task helper
        
        Args:
            taskname (str): Name of the task
            style (str): Style template to use
            
        Returns:
            SigStageHelper: Appropriate subclass for task
            
        Raises:
            RuntimeError: For unsupported tasks
        """
        if taskname == "arlsat":
            return SigArLSATTaskHelper(style)
        else:
            raise RuntimeError("Not Implemented Yet")


class SigArLSATTaskHelper(SigStageHelper):
    """Helper for ArLSAT signature generation
    
    This class handles prompt creation for the ArLSAT logical reasoning task
    in the signature generation stage.
    
    Attributes:
        CHOICE_IDX (list): Identifiers for multiple-choice options
        CODE_HEADER (str): Header for code blocks
        CODE_BLOCK_COMMENT (str): Comment delimiter for code blocks
    """
    
    CHOICE_IDX = ["(A)", "(B)", "(C)", "(D)", "(E)"]
    CODE_HEADER = "### write python code to answer the question"
    CODE_BLOCK_COMMENT = '"""'

    style_to_completion_length = {
        "sigtpl": 2048,
    }

    style_to_train_sep = {
        "sigtpl": "\n\n\n\n",
    }

    def prompt_func(self, test_ex, shots):
        """Create prompts based on style
        
        Args:
            test_ex: Test example to create prompt for
            shots: Shot examples to include in prompt
            
        Returns:
            str: Formatted prompt
            
        Raises:
            RuntimeError: For unsupported styles
        """
        if self.style == "sigtpl":
            return self.sigtpl_prompt(test_ex, shots)
        else:
            raise RuntimeError("Not Implemented Yet")

    def _single_ex_func(self, ex, is_train):
        """Format a single example
        
        Args:
            ex (dict): The example to format
            is_train (bool): Whether this is a training example
            
        Returns:
            str: Formatted example
        """
        assert not is_train
        choice_str = "\n".join([self.CHOICE_IDX[i] + " " + x for (i, x) in enumerate(ex["choices"])])
        p_ex = "{}\n{}\n{}\nQuestion: {}\nChoices:\n{}\n{}\n".format(
            self.CODE_HEADER,
            self.CODE_BLOCK_COMMENT,
            ex["context"], ex["question"], choice_str,
            self.CODE_BLOCK_COMMENT)
        return p_ex

    def sigtpl_prompt(self, test_ex, shots):
        """Create a prompt using the sigtpl style
        
        Args:
            test_ex: Test example to create prompt for
            shots: Shot examples to include in prompt
            
        Returns:
            str: Formatted prompt with shots and test example
        """
        showcase_examples = [
            self._single_ex_func(s, True) for s in shots
        ]
        test_example = [self._single_ex_func(test_ex, False)]
        return  self.get_train_sep().join(showcase_examples + test_example)


class TransStageHelper(MultiStageTaskHelper):
    """Helper for the translation stage
    
    This class specializes MultiStageTaskHelper for the translation
    stage of the multi-stage process.
    """
    
    @classmethod
    def from_taskname(cls, taskname, style):
        """Factory method to create appropriate task helper
        
        Args:
            taskname (str): Name of the task
            style (str): Style template to use
            
        Returns:
            TransStageHelper: Appropriate subclass for task
            
        Raises:
            RuntimeError: For unsupported tasks
        """
        if taskname == "arlsat":
            return TransArLSATTaskHelper(style)
        else:
            raise RuntimeError("Not Implemented Yet")


class TransArLSATTaskHelper(SigStageHelper):
    """Helper for ArLSAT translation
    
    This class handles prompt creation for the ArLSAT logical reasoning task
    in the translation stage.
    
    Attributes:
        CHOICE_IDX (list): Identifiers for multiple-choice options
        CODE_HEADER (str): Header for code blocks
        CODE_BLOCK_COMMENT (str): Comment delimiter for code blocks
    """
    
    CHOICE_IDX = ["(A)", "(B)", "(C)", "(D)", "(E)"]
    CODE_HEADER = "### write python code to answer the question"
    CODE_BLOCK_COMMENT = '"""'

    style_to_completion_length = {
        "transtpl": 256,
    }

    style_to_train_sep = {
        "transtpl": "\n\n\n\n",
    }

    def prompt_func(self, test_ex, shots):
        """Create prompts based on style
        
        Args:
            test_ex: Test example to create prompt for
            shots: Shot examples to include in prompt
            
        Returns:
            str: Formatted prompt
            
        Raises:
            RuntimeError: For unsupported styles
        """
        if self.style == "transtpl":
            return self.transtpl_prompt(test_ex, shots)
        else:
            raise RuntimeError("Not Implemented Yet")

    def _single_ex_func(self, ex, is_train):
        """Format a single example
        
        Args:
            ex (dict): The example to format
            is_train (bool): Whether this is a training example
            
        Returns:
            str: Formatted example
        """
        assert not is_train
        choice_str = "\n".join([self.CHOICE_IDX[i] + " " + x for (i, x) in enumerate(ex["choices"])])
        p_ex = "{}\n{}\n{}\nQuestion: {}\nChoices:\n{}\n{}\n".format(
            self.CODE_HEADER,
            self.CODE_BLOCK_COMMENT,
            ex["context"], ex["question"], choice_str,
            self.CODE_BLOCK_COMMENT)
        return p_ex

    def transtpl_prompt(self, test_ex, shots):
        """Create a prompt using the transtpl style
        
        Args:
            test_ex: Test example to create prompt for
            shots: Shot examples to include in prompt
            
        Returns:
            str: Formatted prompt with shots and test example
        """
        showcase_examples = [
            self._single_ex_func(s, True) for s in shots
        ]
        test_example = [self._single_ex_func(test_ex, False)]
        return  self.get_train_sep().join(showcase_examples + test_example)


class SignatureInfo:
    """Analyzes generated signature code for key components
    
    This class extracts and categorizes Z3 constructs from signature code.
    
    Attributes:
        completion (str): The signature code to analyze
        style_template (str): Style template used for generation
        keywords (set): Set of identified Z3 constructs
    """
    
    def __init__(self, completion, style_template):
        """Initialize with a completion and template
        
        Args:
            completion (str): The signature code to analyze
            style_template (str): Style template used for generation
        """
        self.completion = completion
        self.style_template = style_template
        self.keywords = self.extract_keywords(completion)


    def extract_keywords(self, completion):
        """Identifies critical Z3 elements in the code
        
        Analyzes the signature code to find important Z3 constructs
        like enums, functions, constraints, etc.
        
        Args:
            completion (str): The signature code to analyze
            
        Returns:
            set: Set of found Z3 constructs
        """
        lines = [x.strip() for x in completion.split("\n")]
        decl_lines = [x for x in lines if "EnumSort" in x or "Function" in x]
        print_lines = [x for x in lines if "print" in x]
        question_line = next((x for x in lines if "# Question" in x), "")

        keywords = set()

        enum_types = {}
        for line in decl_lines:
            if "EnumSort" in line:
                sort_name = line.split("=")[0].strip()
                sort_member_str = line.split("=")[1].strip()[len("EnumSort("):-1]
                sort_members = [x.strip() for x in sort_member_str[1:-1].split(",")]
                if all([x.isdigit() for x in sort_members]):
                    enum_types[sort_name] = "EnumInt"
                    keywords.add(enum_types[sort_name])
                else:
                    enum_types[sort_name] = "EnumVal"

            elif "Function" in line:
                function_args_str = line.split("=")[1].strip()[len("Function("):-1]
                function_args = [x.strip() for x in function_args_str.split(",")]
                function_sig = [enum_types[x] if x in enum_types else x for x in function_args]
                function_sig = "(" + ",".join(function_sig) + ")"
                function_sig = function_sig.replace("EnumInt", "int")
                if "int" in function_sig:
                    keywords.add("int")
                if "bool" in function_sig:
                    keywords.add("bool")
                keywords.add(function_sig)
            else:
                raise RuntimeError("Unknown declaration statement: {}".format(line))

        if " if " in question_line.lower():
            keywords.add("if_question")

        for line in print_lines:
            line = line[len("print("):-1]
            if "exception" in line:
                keywords.add("exception")
                line = line[len("exception("):-1]
            keywords.add(line.strip()[:-2])
        return keywords


class TransSetting:
    """Manages translation parameters and example selection
    
    This class handles configuration for the translation stage, including
    example selection, prompt construction, and output formatting.
    
    Attributes:
        args: Command line arguments
        setting_version (str): Name of the configuration preset
        setting (dict): Configuration parameters
    """
    
    SETTING_TO_MATHOD = {
        "setupsatlm": {
            "question_style": "satlm",
            "selection": "signature",
            "prompt": "satlm",
            "train_sep": "\n\n\n\n",
            "completion_length": 3072,
        },
    }

    def __init__(self, args):
        """Initialize with arguments
        
        Args:
            args: Command line arguments containing translation settings
        """
        self.args = args
        setting_version = args.trans_setting
        self.setting_version = setting_version
        self.setting = self.SETTING_TO_MATHOD[setting_version]

    def get_style_template(self):
        """Get the style template for question formatting
        
        Returns:
            str: Style template name
        """
        return self.setting["question_style"]

    def get_train_sep(self):
        """Get separator for training examples
        
        Returns:
            str: Separator string
        """
        return self.setting["train_sep"]

    def get_completion_length(self):
        """Get expected completion length
        
        Returns:
            int: Maximum token length for completion
        """
        return self.setting["completion_length"]

    def shot_selection(self, test_signature, train_signatures, num_shots):
        """Select exemplar shots based on setting
        
        Args:
            test_signature (SignatureInfo): Signature for test example
            train_signatures (list): List of available training signatures
            num_shots (int): Number of shots to select
            
        Returns:
            list: Indices of selected shots
            
        Raises:
            RuntimeError: For unsupported selection methods
        """
        if self.setting["selection"] == "signature":
            return self.signature_base_shots_selection(test_signature, train_signatures, num_shots)
        else:
            raise RuntimeError("Not Implemented Yet")

    def construct_prompt(self, test_ex, train_annotations):
        """Construct translation prompt based on setting
        
        Args:
            test_ex: Test example
            train_annotations: Annotated training examples
            
        Returns:
            str: Formatted prompt
            
        Raises:
            RuntimeError: For unsupported prompt types
        """
        if self.setting["prompt"] == "satlm":
            return self.predefined_prompt(self.setting["prompt"], test_ex, train_annotations)
        else:
            raise RuntimeError("Not Implemented Yet")

    def encode_question(self, test_ex):
        """Encode a question using the specified style
        
        Args:
            test_ex: Test example to encode
            
        Returns:
            str: Encoded question
            
        Raises:
            RuntimeError: For unsupported question styles
        """
        if self.setting["question_style"] == "satlm":
            return self.satlm_encode_question(test_ex)
        else:
            raise RuntimeError("Not Implemented Yet")

    def satlm_encode_question(self, ex):
        """Encode a question using the SATLM style
        
        Args:
            ex: Example to encode
            
        Returns:
            str: Encoded question
        """
        CHOICE_IDX = ["(A)", "(B)", "(C)", "(D)", "(E)"]
        CODE_HEADER = "### Z3 CODE GENERATION TASK - GENERATE ONLY EXECUTABLE Z3 CODE"
        CODE_BLOCK_COMMENT = '"""'
        choice_str = "\n".join([CHOICE_IDX[i] + " " + x for (i, x) in enumerate(ex["choices"])])
        p_ex = "{}\n\n# ⚠️⚠️⚠️ CRITICAL INSTRUCTIONS ⚠️⚠️⚠️\n# 1. ONLY OUTPUT PURE Z3 SOLVER CODE - NO TEXT EXPLANATIONS\n# 2. DO NOT INCLUDE ANY DIRECT ANSWERS OR SOLUTION STATEMENTS\n# 3. YOUR CODE WILL BE EXECUTED SEPARATELY TO FIND THE ANSWER\n# 4. FOLLOW THE Z3 FORMAT EXACTLY\n\n# FORBIDDEN OUTPUT FORMATS:\n# ❌ \"The answer is A.\" \n# ❌ \"Option C must be true.\"\n# ❌ Any explanation or analysis\n\n# CORRECT OUTPUT FORMAT (ONLY CODE):\n# ✓ from z3 import *\n# ✓ def check_valid():\n# ✓     solver = Solver()\n# ✓     ...\n\n{}\n{}\nQuestion: {}\nChoices:\n{}\n{}\n".format(
            CODE_HEADER,
            CODE_BLOCK_COMMENT,
            ex["context"], ex["question"], choice_str,
            CODE_BLOCK_COMMENT
        )
        return p_ex

    def predefined_prompt(self, predev_version, test_ex, train_annotations):
        """Create a prompt using predefined templates
        
        Args:
            predev_version (str): Preset version to use
            test_ex: Test example
            train_annotations: Annotated training examples
            
        Returns:
            str: Formatted prompt
        """
        showcase_examples = [x[predev_version] for x in train_annotations]
        test_example = [self.encode_question(test_ex)]
        return  self.get_train_sep().join(showcase_examples + test_example)

    # return indexes of the shot
    def signature_base_shots_selection(self, test_signature, train_signatures, num_shots):
        """Select shots based on signature similarity
        
        Chooses examples that cover the most keywords in the test signature,
        prioritizing those with higher coverage ratios.
        
        Args:
            test_signature (SignatureInfo): Signature for test example
            train_signatures (list): List of available training signatures
            num_shots (int): Number of shots to select
            
        Returns:
            list: Indices of selected shots
        """
        # try to cover as many keywords as possible
        full_keywords = set(test_signature.keywords)
        remaining_keywords = set(test_signature.keywords)

        selected_indexes = []
        for _ in range(num_shots):
            # max_full_gain = (-1, -1)
            # max_rem_gain = (-1, -1)
            max_gain = ((-1, -1, -1, -1), -1)
            for i, train_signature in enumerate(train_signatures):
                if i in selected_indexes:
                    continue
                rem_gain = len(remaining_keywords.intersection(train_signature.keywords))
                rem_gain_ratio = rem_gain / len(train_signature.keywords)
                full_gain = len(full_keywords.intersection(train_signature.keywords))
                full_gain_ratio = full_gain / len(train_signature.keywords)
                comp_key = (rem_gain, rem_gain_ratio, full_gain, full_gain_ratio)
                if comp_key >= max_gain[0]:
                    max_gain = (comp_key, i)

            selected_indexes.append(max_gain[1])
            remaining_keywords = remaining_keywords.difference(train_signatures[max_gain[1]].keywords)

        return selected_indexes

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
    prompt_lines = read_jsonline(f'manual_prompts/multistage_{task}.jsonline')
    d = dict([(x["id"], x) for x in prompt_lines])
    selected = d[prompt_id]
    assert selected["style_template"] == style_template
    return selected["prompt"]


def sig_stage_result_filename_func(args):
    """Generates cache filenames for signature stage
    
    Args:
        args: Command line arguments
        
    Returns:
        str: Formatted cache filename
        
    Pattern: 
        multisgate-sig-[task]--[split]--[slice_range]--[prompt_id]--[params]
    """
    if args.sig_method == "manual":
        prompt_id = "manual" + args.sig_prompt_id
    else:
        raise RuntimeError("Not Implemented Yet")

    model_suffix = f"--model{args.model}" if hasattr(args, "model") else ""

    return "misc/multisgate-sig-{}--{}{}-{}--{}--numsamp{}--temp{}--sty{}{}--predictions.json".format(
        args.task,
        get_eval_split_abbrev(args),
        args.slice_dev, args.slice_dev + args.num_dev,
        prompt_id,
        args.num_samples,
        args.temperature,
        args.sig_style_template,
        model_suffix
    )

def trans_stage_result_filename_func(args):
    """Generates unique filename for translation stage results cache
    
    Args:
        args: Runtime configuration
        
    Returns:
        str: Filename path for caching
        
    Format:
        misc/multisgate-trans-{task}--eng{engine}--{eval_split}{slice_dev}-{slice_dev+num_dev}--sig{sig_p_id}--st{trans_setting}--{num_trans_shots}--numsamp{num_samples}--temp{temperature}--sty{sig_style_template}--predictions.json
    """
    sig_p_id = args.sig_prompt_id

    if not sig_p_id:
        raise RuntimeError("Not Implemented Yet")
        
    model_suffix = f"--model{args.model}" if hasattr(args, "model") else ""
        
    return "misc/multisgate-trans-{}--eng{}--{}{}-{}--sig{}--st{}--{}--numsamp{}--temp{}--sty{}{}--predictions.json".format(
        args.task,
        "",  # Removed args.engine reference
        get_eval_split_abbrev(args),
        args.slice_dev, args.slice_dev + args.num_dev,
        sig_p_id,
        args.trans_setting,
        args.num_trans_shots,
        args.num_samples,
        args.temperature,
        args.sig_style_template,
        model_suffix
    )


def parse_problem_signatures(args, responses, task_helper):
    """Converts LM outputs to SignatureInfo objects
    
    Args:
        args: Command line arguments
        responses: Raw LM completion texts
        task_helper: Helper for signature processing
        
    Returns:
        list: List of analyzed signature objects
    """
    signatures = []
    for reps in responses:
        sigs = []
        for r in reps:
            completion = r["text"].strip()
            sig = SignatureInfo(completion, args.sig_style_template)
            sigs.append(sig)

        signatures.append(sigs)

    return signatures

def run_signature_stage(args, test_data):
    """Generates formal problem signatures using LM
    
    Args:
        args: Command line arguments
        test_data: List of problem instances
        
    Returns:
        list: Problem signatures as SignatureInfo objects
        
    Process:
        1. Constructs signature prompts
        2. Executes LM completion
        3. Parses output signatures
    """
    print("\n")
    print(f"STARTING SIGNATURE GENERATION - Processing {len(test_data)} examples")
    print(f"SIGNATURE METHOD: {args.sig_method}")
    print(f"SIGNATURE PROMPT ID: {args.sig_prompt_id}")
    print("="*50)
    
    task_helper = SigStageHelper.from_taskname(args.task, args.sig_style_template)

    # construct signature prompt
    if args.sig_method == "manual":
        base_manual_prompt = read_manual_prompt(args.task, SIG_STAGE, args.sig_prompt_id, args.sig_style_template)
    else:
        raise RuntimeError("Not Implemented Yet")

    prompts_to_complete = []    
    for idx, test_ex in enumerate(test_data):
        print(f"SIGNATURE: Preparing example {idx}/{len(test_data)}")
        test_part = task_helper.prompt_func(test_ex, [])
        
        prompts_to_complete.append(
            [base_manual_prompt + task_helper.get_train_sep() + test_part]
        )

    _batch_size, _temperature, _num_samples = args.batch_size, args.temperature, args.num_samples
    args.batch_size, args.temperature, args.num_samples = 5, 0.0, 1
    task_max_tokens = task_helper.get_completion_length()
    task_stop_token = task_helper.get_train_sep()
    cache_filename = sig_stage_result_filename_func(args)
    
    print("\n")
    print(f"SIGNATURE: Running LM completion for {len(prompts_to_complete)} examples")
    print(f"SIGNATURE: Cache file: {cache_filename}")
    print("="*50)
    
    responses = run_completion_tasks_with_cache(args, cache_filename, prompts_to_complete, task_max_tokens, task_stop_token)
    responses = [flatten_nested_list(resps_by_example) for resps_by_example in responses]
    args.batch_size, args.temperature, args.num_samples = _batch_size, _temperature, _num_samples

    print("\n")
    print("SIGNATURE: Parsing output signatures")
    print("="*50)
    # signature stage evaluation
    problem_signatures = parse_problem_signatures(args, responses, task_helper)
    
    print("SIGNATURE GENERATION COMPLETED")
    print("="*50)
    return problem_signatures


TASK_ANNOTATION_DICT = {
    "arlsat": ["signature", "satlm",],
}

def read_trans_annotations(args):
    """Loads human-annotated translation examples
    
    Args:
        args: Command line arguments
        
    Returns:
        list: Annotated examples with problem, signature, and solution
    """
    prefix = join(TRANS_ANNOTATION_DIR, args.task)

    annotation_list = TASK_ANNOTATION_DICT[args.task]

    annotations = []
    ex_names = [x for x in os.listdir(prefix) if not x.startswith(".")]
    ex_names = sorted(ex_names, key=lambda x: int(re.findall(r"\d+", x)[-1]))

    for ex_name in ex_names:
        if ex_name.startswith("."):
            continue
        anno = {}
        anno["name"] = ex_name
        for fname in annotation_list:
            if os.path.exists(join(prefix, ex_name, fname + ".py")):
                with open(join(prefix, ex_name, fname + ".py")) as f:
                    anno[fname] = f.read()
            else:
                anno[fname] = None
        annotations.append(anno)

    return annotations


def strip_question_head(x):
    """Extract question from a formatted string
    
    Args:
        x (str): Formatted question string
        
    Returns:
        str: Extracted question
    """
    return x.split('"""')[-1].strip()


def run_translation_stage(args, test_data, problem_signatures):
    """Converts signatures to executable Z3 code
    
    Args:
        args: Command line arguments
        test_data: List of problem instances
        problem_signatures: Output from signature stage
        
    Process:
        1. Selects relevant training examples
        2. Generates code prompts
        3. Executes code generation
        4. Evaluates outputs
    """
    print("\n")
    print(f"STARTING TRANSLATION STAGE - Processing {len(test_data)} examples")
    print(f"TRANSLATION SETTING: {args.trans_setting}")
    print(f"TRANSLATION SHOTS: {args.num_trans_shots}")
    print("="*50)
    
    sig_helper = SigStageHelper.from_taskname(args.task, args.sig_style_template)
    trans_setting = TransSetting(args)

    print("TRANSLATION: Loading annotations")
    train_example_annotations = read_trans_annotations(args)
    for ex_ann in train_example_annotations:
        ex_ann["sig_info"] = SignatureInfo(strip_question_head(ex_ann["signature"]), args.sig_style_template)

    # If resuming, check if partial results exist
    cache_filename = trans_stage_result_filename_func(args)
    partial_cache_file = cache_filename + ".partial"
    cached_results = None
    processed_examples = []
    
    print(f"TRANSLATION: Cache file will be {cache_filename}")
    
    if args.resume and os.path.exists(partial_cache_file):
        try:
            print(f"TRANSLATION: Found partial results at {partial_cache_file}")
            print(f"TRANSLATION: Attempting to resume from previous run")
            cached_results = read_json(partial_cache_file)
            processed_examples = [i for i, results in enumerate(cached_results) if results]
            
            if processed_examples:
                print(f"TRANSLATION: Will resume from example {max(processed_examples) + 1}")
                print(f"TRANSLATION: Skipping {len(processed_examples)} already processed examples")
            else:
                print(f"TRANSLATION: No complete examples found, starting from beginning")
            
            # Print which examples already have saved responses
            for idx in processed_examples:
                print(f"TRANSLATION: ✓ ALREADY COMPLETED: Example {idx}")
                
        except Exception as e:
            print(f"TRANSLATION: Error reading partial results: {e}")
            print(f"TRANSLATION: Starting from beginning")
            processed_examples = []
            cached_results = None
    elif args.resume:
        print(f"TRANSLATION: No partial results found at {partial_cache_file}")
        print(f"TRANSLATION: Starting from beginning")
    else:
        print(f"TRANSLATION: Resume flag not set, starting from beginning")
    
    # If all examples are already processed, just use the cached results
    if args.resume and cached_results and len(processed_examples) == len(test_data):
        print("\n")
        print("TRANSLATION: All examples already processed, using cached results")
        print("="*50)
        responses = cached_results
    else:
        # Create prompts for examples that need processing
        prompts_to_complete = []
        to_process = []
        for idx, (test_ex, test_sigs) in enumerate(zip(test_data, problem_signatures)):
            if args.resume and idx in processed_examples:
                print(f"TRANSLATION: SKIPPING Example {idx} (already processed)")
                continue
                
            print(f"TRANSLATION: PREPARING Example {idx}")
            to_process.append(idx)
            prompts_for_ex = []
            for test_sig in test_sigs:
                selected_indexes = trans_setting.shot_selection(test_sig, [x["sig_info"] for x in train_example_annotations], args.num_trans_shots)
                selected_annotations = [train_example_annotations[i] for i in selected_indexes]

                prompt = trans_setting.construct_prompt(test_ex, selected_annotations)
                prompts_for_ex.append(prompt)
            
            prompts_to_complete.append(prompts_for_ex)
        
        print("\n")
        if to_process:
            print(f"TRANSLATION: Will process examples: {to_process}")
        else:
            print(f"TRANSLATION: No examples to process")
        print("="*50)
        
        # Process only unprocessed examples
        if prompts_to_complete:
            task_max_tokens = trans_setting.get_completion_length()
            task_stop_token = trans_setting.get_train_sep()
            
            print("\n")
            print(f"TRANSLATION: Running LM completion for {len(prompts_to_complete)} examples")
            print(f"TRANSLATION: Examples being processed: {to_process}")
            print("="*50)
            
            new_responses = run_completion_tasks_with_cache(args, cache_filename, prompts_to_complete, task_max_tokens, task_stop_token)
            
            # Merge new responses with cached ones
            if args.resume and cached_results:
                print("\n")
                print("TRANSLATION: Merging new responses with cached results")
                print("="*50)
                
                responses = cached_results.copy()
                # Update with new responses
                new_idx = 0
                for idx in range(len(test_data)):
                    if idx not in processed_examples:
                        # Replace empty results with new ones
                        responses[idx] = new_responses[new_idx]
                        print(f"TRANSLATION: 💾 SAVED Example {idx}")
                        new_idx += 1
            else:
                responses = new_responses
                # Print saving information for all responses
                print("\n")
                print("TRANSLATION: Saving all responses")
                print("="*50)
                for idx in range(len(responses)):
                    print(f"TRANSLATION: 💾 SAVED Example {idx}")
        else:
            # No new examples to process
            responses = cached_results
    
    print("\n")
    print("TRANSLATION: Flattening and evaluating responses")
    print("="*50)
    # Flatten responses for evaluation
    responses = [flatten_nested_list(resps_by_example) for resps_by_example in responses]

    args.style_template = trans_setting.get_style_template()
    eval_results = run_evaluation(args, test_data, responses)
    
    print("\n")
    print("TRANSLATION STAGE COMPLETED")
    print("="*50)

def multistage_prompting(args):
    """Orchestrates the two-stage reasoning pipeline
    
    Flow:
        1. Signature Generation: Create formal problem representations
        2. Translation: Convert signatures to executable code
        
    Args:
        args: Configured runtime parameters
    
    Calls:
        run_signature_stage() -> run_translation_stage()
    """
    _, test_data = load_train_test_set(args)
    print("\n")
    print("="*80)
    print("STARTING MULTI-STAGE PROMPTING PIPELINE")
    print(f"Total examples in dataset: {len(test_data)}")
    print(f"RESUME FLAG: {args.resume}")
    print(f"FORCE OVERRIDE FLAG: {getattr(args, 'force_override', 'NOT SET')}")
    print("="*80)
    print("\n")

    # Check if signature cache exists and if we're resuming
    sig_cache_filename = sig_stage_result_filename_func(args)
    if args.resume and os.path.exists(sig_cache_filename) and not args.force_override:
        print("="*80)
        print("STAGE 1: SIGNATURE GENERATION - SKIPPING (USING CACHED SIGNATURES)")
        print("="*80)
        # Load existing signatures
        responses = read_json(sig_cache_filename)
        responses = [flatten_nested_list(resps_by_example) for resps_by_example in responses]
        problem_signatures = parse_problem_signatures(args, responses, SigStageHelper.from_taskname(args.task, args.sig_style_template))
        print(f"Loaded {len(problem_signatures)} problem signatures from cache")
    else:
        print("="*80)
        print("STAGE 1: SIGNATURE GENERATION")
        print(f"Reason for not skipping: resume={args.resume}, file_exists={os.path.exists(sig_cache_filename)}, not_force_override={not getattr(args, 'force_override', False)}")
        print("="*80)
        problem_signatures = run_signature_stage(args, test_data)
    
    print("\n")
    print("="*80)
    print("STAGE 2: TRANSLATION")
    print("="*80)
    run_translation_stage(args, test_data, problem_signatures)
    
    print("\n")
    print("="*80)
    print("MULTI-STAGE PIPELINE COMPLETED")
    print("="*80)

def main():
    """Main entry point for the multi-stage prompting system
    
    Process:
        1. Parse arguments
        2. Configure API
        3. Run multi-stage prompting
    """
    parser = argparse.ArgumentParser()
    register_base_args(parser)
    register_multistage_args(parser)

    args = parser.parse_args()
    assert args.task is not None

    config_args_and_api(args)
    multistage_prompting(args)

if __name__=="__main__":
    main()
