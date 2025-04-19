"""
Task evaluation module for assessing model performance on reasoning tasks.

This module provides evaluators for different reasoning tasks, including
methods for parsing completions, extracting answers, computing evaluation
metrics, and comparing predictions to ground truth.
"""
import os
import argparse
import sys
import json
import re
import random
import func_timeout
from func_timeout import FunctionTimedOut
import numpy as np
from tqdm import tqdm

from abc import ABC
from collections import namedtuple, Counter, OrderedDict
from os.path import join

from prog_solver.gsm_solver import  gsm_proglm_exec, gsm_satlm_exec
from prog_solver.clutrr_solver import clutrr_proglm_exec, clutrr_satlm_exec
from prog_solver.proof_solver import proof_proglm_exec, proof_satlm_exec
from prog_solver.arlsat_solver import arlsat_satlm_exec
from prog_solver.boardgame_solver import board_satlm_exec


EVALUATOR_REGISTRY = {}

Prediction = namedtuple('Prediction', ['completion', 'prompt', 'logprob', 'norm_logprob'])

def print_tabular_results(row_id, eval_result):
    """Print evaluation results in a tabular format.
    
    Args:
        row_id: Identifier for the evaluation row
        eval_result (dict): Evaluation metrics
    """
    num_contents = [ "%.2f" % (eval_result["accuracy"] * 100), "%.2f" % (eval_result["consistency"] * 100),
        str(eval_result["avg_logprob"]), str(eval_result["avg_normlogprob"])]
    print("\t".join(["TABINFO", str(row_id)] + num_contents))

class TaskEvaluator(ABC):
    """Base class for task-specific evaluators.
    
    This abstract class provides the interface and shared functionality
    for evaluating model outputs on reasoning tasks. Task-specific evaluators
    should inherit from this class and implement its abstract methods.
    
    Attributes:
        do_printing (bool): Whether to print detailed outputs
        do_impose_prediction (bool): Whether to impose predictions when none are found
        do_voting (bool): Whether to use voting among multiple samples
        NULL_ANSWER (str): Placeholder for null/missing answers
        EXCEPTION (str): Placeholder for exceptions during evaluation
        TIMEOUT (str): Placeholder for timeouts during evaluation
        AMBIG (str): Placeholder for ambiguous answers
        UNSAT (str): Placeholder for unsatisfiable problems
    """
    do_printing = False
    do_impose_prediction = False
    do_voting = False
    NULL_ANSWER = "NULL"
    EXCEPTION = "EXCEPTION"
    TIMEOUT = "TIMEOUT"
    AMBIG = "AMBIG"
    UNSAT = "UNSAT"

    @classmethod
    def get_task_name(cls):
        """Extract task name from the evaluator class name.
        
        Returns:
            str: Lowercase task name
        """
        [task_name] = re.match("(.+)Evaluator", cls.__name__).groups()
        return task_name.lower()

    def __init_subclass__(cls, **kwargs):
        """Register all children in registry.
        
        This automatically registers all subclasses in the EVALUATOR_REGISTRY.
        """
        super().__init_subclass__(**kwargs)
        if cls == TaskEvaluator:
            # print(f"{cls} is abstract!")
            return
        task_name = cls.get_task_name().lower()
        EVALUATOR_REGISTRY[task_name] = cls

    @classmethod
    def process_instance(cls, pred, ref, prompting_style=None):
        """Process a single instance's predictions.
        
        Args:
            pred (list): Predictions for an instance
            ref (dict): Reference/ground truth for an instance
            prompting_style (str, optional): Style of prompting used
            
        Returns:
            dict: Processed instance with answers and explanations
        """
        choices = []
        gt = cls.postprocess_ground_truth(ref["label"])
        null_ans = cls.NULL_ANSWER
        prompt = pred[0].prompt
        for p in pred:
            single_comp, single_exp, single_ans = cls.parse_explanation_answer_from_completion(p.completion, prompting_style)
            choices.append({
                "completion": single_comp,
                "answer": single_ans,
                "explanation": single_exp,
                "norm_logprob": p.norm_logprob,
                "sum_logprob": p.logprob,
                "acc": str(gt == single_ans),
            })

        return {
            "prompt": prompt,
            "ground_truth": gt,
            "null_answer": null_ans,
            "completions": choices
        }

    @classmethod
    def enter_evaluation(cls):
        """Perform setup before starting evaluation.
        
        This method is called before evaluation begins and can be
        overridden by subclasses to perform task-specific setup.
        """
        pass

    @classmethod
    def exit_evaluation(cls):
        """Perform cleanup after finishing evaluation.
        
        This method is called after evaluation completes and can be
        overridden by subclasses to perform task-specific cleanup.
        """
        pass

    @classmethod
    def generate_random_answer(cls):
        """Generate a random answer when needed.
        
        This is used when do_impose_prediction is True and no answer is found.
        
        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        raise NotImplementedError

    @classmethod
    def evaluate(cls, predictions, examples, prompting_style=None, train_sep="\n\n", return_verbose=False):
        """Evaluate predictions against ground truth examples.
        
        Args:
            predictions (list): Model predictions
            examples (list): Ground truth examples
            prompting_style (str, optional): Style of prompting used
            train_sep (str, optional): Separator for training examples
            return_verbose (bool, optional): Whether to return detailed results
            
        Returns:
            dict: Evaluation metrics including accuracy and consistency
        """
        if isinstance(predictions[0], list) and len(predictions[0]) > 1:
            cls.do_voting = True

        cls.enter_evaluation()

        acc_records = []
        cov_records = []
        cons_records = []

        all_proced_answers = []
        all_proced_gts = []
        all_voted_answers = []

        for idx, (pred, ref) in tqdm(enumerate(zip(predictions, examples)), total=max(len(predictions), len(examples)), desc="Evaluating"):
            if isinstance(pred, list):
                all_answers = []
                comp = []
                prompt = cls.postprocess_prompt(pred[0].prompt, train_sep)
                answer_counter = {}
                for p in pred:
                    single_comp, single_ans = cls.postprocess_completion(p.completion, prompting_style, train_sep, example=ref)
                    all_answers.append(single_ans)
                    comp.append(single_comp)
                    if single_ans not in answer_counter:
                        answer_counter[single_ans] = {
                            "count": 0,
                            "max_logprob": -1e6,
                            "max_norm_logprob": -1e6,
                        }
                    stat = answer_counter[single_ans]
                    stat["count"] = stat["count"] + 1
                    stat["max_logprob"] = max(stat["max_logprob"], p.logprob)
                    stat["max_norm_logprob"] = max(stat["max_norm_logprob"], p.norm_logprob)

                sorted_answers = sorted(answer_counter.keys(), key=lambda x: (answer_counter[x]["count"], answer_counter[x]["max_norm_logprob"]), reverse=True)
                # sorted_answers = sorted(answer_counter.keys(), key=lambda x: ( answer_counter[x]["max_norm_logprob"],answer_counter[x]["count"] ), reverse=True)
                answer = sorted_answers[0]
                if answer == cls.NULL_ANSWER and len(sorted_answers) > 1:
                    answer = sorted_answers[1]
                if cls.NULL_ANSWER in sorted_answers:
                    sorted_answers.remove(cls.NULL_ANSWER)
                cons = answer_counter[answer]['count'] / len(pred)
                answer_counter = OrderedDict([(k, answer_counter[k]) for k in sorted_answers])
            else:
                prompt = cls.postprocess_prompt(pred.prompt)
                comp, answer = cls.postprocess_completion(pred.completion, prompting_style, train_sep, example=ref)
                cons = 1.0
                answer_counter = None
                all_answers = [answer]
            if answer == cls.NULL_ANSWER and cls.do_impose_prediction:
                answer = cls.generate_random_answer()

            gt = cls.postprocess_ground_truth(ref["label"])
            acc_records.append(cls.answer_equal(answer, gt, example=ref))
            cons_records.append(cons)
            if answer_counter is not None:
                cov_records.append(gt in answer_counter)
            all_proced_answers.append(all_answers)
            all_voted_answers.append(answer)
            all_proced_gts.append(gt)
            cls.print_instance_outputs(idx, prompt, comp, answer, gt, ref, answer_counter)

        eval_results = {}
        acc_records = np.array(acc_records)
        print("ACC: {:.2f}".format(np.mean(acc_records) * 100))
        # print("CONS: {:.2f}".format(np.mean(cons_records) * 100))
        eval_results["accuracy"] = np.mean(acc_records)
        eval_results["consistency"] = np.mean(cons_records)
        if cov_records:
            cov_records = np.array(cov_records)
            # print("COV: {:.2f}".format(np.mean(cov_records) * 100))
            eval_results["converage"] = np.mean(cov_records)
        if return_verbose:
            eval_results["all_raw_predictions"] = all_proced_answers
            eval_results["all_gts"] = all_proced_gts
            eval_results["all_voted_predictions"] = all_voted_answers
        eval_results["num"] = len(acc_records)

        cls.exit_evaluation()
        return eval_results

    @staticmethod
    def answer_equal(pred, gt, example=None):
        """Compare prediction to ground truth.
        
        Args:
            pred: Predicted answer
            gt: Ground truth answer
            example (dict, optional): Example context
            
        Returns:
            bool: True if answers are equal, False otherwise
        """
        return pred == gt

    @classmethod
    def print_instance_outputs(cls, idx, prompt, comp, answer, gt, ref, answer_counter=None):
        """Print detailed outputs for a single instance.
        
        Args:
            idx (int): Instance index
            prompt (str): Input prompt
            comp (str): Model completion
            answer (str): Extracted answer
            gt (str): Ground truth answer
            ref (dict): Reference example
            answer_counter (dict, optional): Counter of answers
        """
        # Only executed when do_printing is True
        print(f"========= Example {idx} =========")
        print("Prompt:", prompt)
        if isinstance(comp, list):
            for (c, a) in zip(comp, answer):
                print("============= COMPLETION =============")
                print(c)
                print("=========== END COMPLETION ===========")
                print("Answer:", a, "Ground truth:", gt, "Correct?", a == gt)
        else:
            print("============= COMPLETION =============")
            print(comp)
            print("=========== END COMPLETION ===========")
            print("Answer:", answer, "Ground truth:", gt, "Correct?", answer == gt)
        if answer_counter:
            print("Answer counter:", answer_counter)
        print("============================")

    @classmethod
    def core_evaluation(cls, predictions, examples, prompting_style=None):
        """Core evaluation logic for task-specific implementations.
        
        Args:
            predictions (list): Model predictions
            examples (list): Ground truth examples
            prompting_style (str, optional): Style of prompting used
            
        Returns:
            tuple: Evaluation results and processed data
        """
        pass

    @staticmethod
    def postprocess_completion(completion, prompting_style, train_sep, example=None):
        """Process raw model completion into a standard format.
        
        Args:
            completion (str): Raw model completion
            prompting_style (str): Style of prompting used
            train_sep (str): Separator for training examples
            example (dict, optional): Context example
            
        Returns:
            tuple: Processed completion and extracted answer
        """
        raise NotImplementedError

    @staticmethod
    def postprocess_ground_truth(gt):
        """Process ground truth into standard format.
        
        Args:
            gt: Raw ground truth
            
        Returns:
            Standardized ground truth
        """
        return gt

    @classmethod
    def parse_explanation_answer_from_completion(cls, completion, prompting_style):
        """Parse explanation and answer from model completion.
        
        Args:
            completion (str): Model completion
            prompting_style (str): Style of prompting used
            
        Returns:
            tuple: (completion, explanation, answer)
        """
        comp, ans = cls.postprocess_completion(completion, prompting_style, "\n\n")
        return comp, comp, ans

    @staticmethod
    def postprocess_prompt(prompt, train_sep):
        """Process prompt into standard format.
        
        Args:
            prompt (str): Raw prompt
            train_sep (str): Separator for training examples
            
        Returns:
            str: Processed prompt
        """
        return prompt.split(train_sep)[-1]

class GSMEvaluator(TaskEvaluator):
    """Evaluator for Grade School Math (GSM) problems.
    
    This evaluator handles numerical answers from math word problems,
    supporting multiple answer formats and prompting styles.
    """
    ANSWER_RE = re.compile(r"(\-?[0-9\.\,]+)")
    ANSWER_HINT = "the answer is"

    GSM_ERROR_ANSWER = [
        TaskEvaluator.UNSAT, TaskEvaluator.EXCEPTION, TaskEvaluator.TIMEOUT, TaskEvaluator.AMBIG
    ]

    @staticmethod
    def postprocess_ground_truth(gt):
        """Standardize ground truth format for GSM problems.
        
        Args:
            gt: Raw ground truth answer
            
        Returns:
            str: Standardized ground truth
        """
        return gt.strip()

    @staticmethod
    def answer_equal(pred, gt, example=None):
        """Compare numerical answers, handling formatting variations.
        
        Args:
            pred (str): Predicted answer
            gt (str): Ground truth answer
            example (dict, optional): Example context
            
        Returns:
            bool: True if answers are numerically equal, False otherwise
        """
        if pred in [TaskEvaluator.NULL_ANSWER, TaskEvaluator.EXCEPTION, TaskEvaluator.TIMEOUT]:
            return False
        try:
            return float(pred.replace(",", "")) == float(gt.replace(",", ""))
        except:
            return False

    @staticmethod
    def postprocess_completion(completion, prompting_style, train_sep, example=None):
        """Process GSM problem completions based on prompting style.
        
        Args:
            completion (str): Raw model completion
            prompting_style (str): Style of prompting used
            train_sep (str): Separator for training examples
            example (dict, optional): Context example
            
        Returns:
            tuple: Processed completion and extracted answer
        """
        completion = completion.rstrip().split(train_sep)[0]
        if prompting_style in ["cot", "direct"]:
            return GSMEvaluator.postprocess_qa_style_completion(completion)
        elif prompting_style == "program":
            return GSMEvaluator.postprocess_prog_style_completion(completion)
        elif prompting_style in ["sat", "satlm"]:
            return GSMEvaluator.postprocess_sat_style_completion(completion, prompting_style)
        else:
            raise RuntimeError("Not implemented")

    @staticmethod
    def postprocess_qa_style_completion(completion):
        """Process completions in question-answering style.
        
        Args:
            completion (str): Raw completion
            
        Returns:
            tuple: Processed completion and extracted answer
        """
        completion = completion.strip()
        ans = GSMEvaluator.extract_answer(completion)
        if ans is None:
            ans = GSMEvaluator.NULL_ANSWER
        return completion, ans

    @staticmethod
    def postprocess_prog_style_completion(completion):
        """Process completions in programming style.
        
        Args:
            completion (str): Raw completion
            
        Returns:
            tuple: Processed completion and extracted answer
        """
        completion = completion.strip()
        try:
            _, answer = gsm_proglm_exec(completion, False)
            answer = answer.strip()
            if "ExecutionError" in answer:
                answer = GSMEvaluator.NULL_ANSWER
        except:
            answer = GSMEvaluator.NULL_ANSWER
        return completion, answer

    @staticmethod
    def postprocess_sat_style_completion(completion, prompting_style):
        """Process completions in SAT-style.
        
        Args:
            completion (str): Raw completion
            prompting_style (str): Prompting style used
            
        Returns:
            tuple: Processed completion and extracted answer
        """
        completion = completion.strip()
        try:
            _, answer = gsm_satlm_exec(completion, False)
            answer = answer.strip()
            if "ExecutionError" in answer:
                answer = GSMEvaluator.NULL_ANSWER
        except:
            answer = GSMEvaluator.NULL_ANSWER
        return completion, answer

    @staticmethod
    def extract_answer(completion):
        """Extract numerical answer from text completion.
        
        Args:
            completion (str): Text completion
            
        Returns:
            str: Extracted numerical answer or None if not found
        """
        if GSMEvaluator.ANSWER_HINT in completion.lower():
            result = completion.lower().strip().split(GSMEvaluator.ANSWER_HINT)[1].strip().rstrip(".")
            m = GSMEvaluator.ANSWER_RE.search(result)
            if m is not None:
                [num] = m.groups()
                return num
        return None

class CLUTRREvaluator(TaskEvaluator):
    """Evaluator for CLUTRR reasoning tasks.
    
    This evaluator handles relation inference tasks, supporting
    different prompting styles and answer formats.
    """
    @staticmethod
    def postprocess_ground_truth(gt):
        """Standardize ground truth for CLUTRR tasks.
        
        Args:
            gt: Raw ground truth
            
        Returns:
            str: Standardized ground truth
        """
        return gt

    @staticmethod
    def postprocess_completion(completion, prompting_style, train_sep, example=None):
        """Process CLUTRR completions based on prompting style.
        
        Args:
            completion (str): Raw model completion
            prompting_style (str): Style of prompting used
            train_sep (str): Separator for training examples
            example (dict, optional): Context example
            
        Returns:
            tuple: Processed completion and extracted answer
        """
        completion = completion.rstrip().split(train_sep)[0]
        if prompting_style in ["cot", "direct"]:
            return CLUTRREvaluator.postprocess_qa_style_completion(completion, prompting_style)
        elif prompting_style == "program":
            return CLUTRREvaluator.postprocess_prog_style_completion(completion)
        elif prompting_style in ["sat", "satlm"]:
            return CLUTRREvaluator.postprocess_sat_style_completion(completion, prompting_style)
        else:
            raise RuntimeError("Not implemented")

    @staticmethod
    def postprocess_qa_style_completion(completion, prompting_style):
        """Process completions in question-answering style.
        
        Args:
            completion (str): Raw completion
            prompting_style (str): Prompting style used
            
        Returns:
            tuple: Processed completion and extracted answer
        """
        completion = completion.strip()
        if "the relation is" in completion.lower():
            result = completion.lower().strip().split("the relation is")[1].strip().rstrip(".")
            return completion, result
        if "so the answer is" in completion.lower():
            result = completion.lower().strip().split("so the answer is")[1].strip().rstrip(".")
            return completion, result
        return completion, CLUTRREvaluator.NULL_ANSWER

    @staticmethod
    def postprocess_prog_style_completion(completion):
        """Process completions in programming style.
        
        Args:
            completion (str): Raw completion
            
        Returns:
            tuple: Processed completion and extracted answer
        """
        completion = completion.strip()
        try:
            _, answer = clutrr_proglm_exec(completion, False)
            answer = answer.strip()
            if "ExecutionError" in answer:
                answer = CLUTRREvaluator.NULL_ANSWER
        except:
            answer = CLUTRREvaluator.NULL_ANSWER
        return completion, answer

    @staticmethod
    def postprocess_sat_style_completion(completion, prompting_style):
        """Process completions in SAT-style.
        
        Args:
            completion (str): Raw completion
            prompting_style (str): Prompting style used
            
        Returns:
            tuple: Processed completion and extracted answer
        """
        completion = completion.strip()
        try:
            _, answer = clutrr_satlm_exec(completion, False)
            answer = answer.strip()
            if "ExecutionError" in answer:
                answer = CLUTRREvaluator.NULL_ANSWER
        except:
            answer = CLUTRREvaluator.NULL_ANSWER
        return completion, answer

class ProofD5Evaluator(TaskEvaluator):
    """Evaluator for D5 proof problems.
    
    This evaluator handles logical proofs, supporting different
    prompting styles and answer formats.
    """
    @staticmethod
    def postprocess_ground_truth(gt):
        """Standardize ground truth for proof problems.
        
        Args:
            gt: Raw ground truth
            
        Returns:
            str: Standardized ground truth
        """
        return gt

    @classmethod
    def enter_evaluation(cls):
        """Set up random seed for reproducibility in proof evaluation."""
        random.seed(42)

    @staticmethod
    def postprocess_completion(completion, prompting_style, train_sep, example=None):
        """Process proof problem completions based on prompting style.
        
        Args:
            completion (str): Raw model completion
            prompting_style (str): Style of prompting used
            train_sep (str): Separator for training examples
            example (dict, optional): Context example
            
        Returns:
            tuple: Processed completion and extracted answer
        """
        completion = completion.rstrip().split(train_sep)[0]
        if prompting_style == "cot":
            return ProofD5Evaluator.postprocess_cot_style_completion(completion)
        elif prompting_style in ["sat", "satlm"]:
            return ProofD5Evaluator.postprocess_sat_style_completion(completion, prompting_style)
        elif prompting_style == "program":
            return ProofD5Evaluator.postprocess_prog_style_completion(completion)
        else:
            raise RuntimeError("Not implemented")

    @staticmethod
    def postprocess_cot_style_completion(completion):
        """Process completions in chain-of-thought style.
        
        Args:
            completion (str): Raw completion
            
        Returns:
            tuple: Processed completion and extracted answer
        """
        completion = completion.strip()
        if "the answer is" in completion.lower():
            result = completion.lower().strip().split("the answer is")[1].strip().rstrip(".")
            return completion, result
        else:
            return completion, ProofD5Evaluator.NULL_ANSWER

    @staticmethod
    def postprocess_sat_style_completion(completion, prompting_style):
        """Process completions in SAT-style.
        
        Args:
            completion (str): Raw completion
            prompting_style (str): Prompting style used
            
        Returns:
            tuple: Processed completion and extracted answer
        """
        completion = completion.strip()
        try:
            _, result = proof_satlm_exec(completion, False)
            result = result.strip()
            if "ExecutionError" in result:
                result = ProofD5Evaluator.NULL_ANSWER
        except:
            result = ProofD5Evaluator.NULL_ANSWER
        return completion, result

    @staticmethod
    def postprocess_prog_style_completion(completion):
        """Process completions in programming style.
        
        Args:
            completion (str): Raw completion
            
        Returns:
            tuple: Processed completion and extracted answer
        """
        completion = completion.strip()
        try:
            _, result = proof_proglm_exec(completion, False)
            result = result.strip()
            if "ExecutionError" in result:
                result = ProofD5Evaluator.NULL_ANSWER
        except:
            result = ProofD5Evaluator.NULL_ANSWER
        return completion, result

    @classmethod
    def generate_random_answer(cls):
        """Generate a random yes/no/unknown answer.
        
        Returns:
            str: Random answer from 'yes', 'no', or 'unknown'
        """
        return random.choice(["yes", "no", "unknown"])

class LongContextMCEvaluator(TaskEvaluator):
    """Evaluator for long-context multiple-choice problems.
    
    This evaluator handles multiple-choice questions with potentially
    long context, supporting different prompting styles.
    """
    ANSWER_HINT = "the answer is"
    CHOICES = ['a', 'b', 'c', 'd', 'e']

    @staticmethod
    def postprocess_ground_truth(gt):
        """Standardize ground truth for multiple-choice problems.
        
        Args:
            gt: Raw ground truth
            
        Returns:
            str: Standardized ground truth
        """
        return gt.lower()

    @classmethod
    def enter_evaluation(cls):
        """Set up random seed for reproducibility in evaluation."""
        random.seed(42)

    @classmethod
    def generate_random_answer(cls):
        """Generate a random multiple-choice answer.
        
        Returns:
            str: Random answer from available choices
        """
        return random.choice(cls.CHOICES)

    @staticmethod
    def postprocess_completion(completion, prompting_style, train_sep, example=None):
        """Process multiple-choice completions based on prompting style.
        
        Args:
            completion (str): Raw model completion
            prompting_style (str): Style of prompting used
            train_sep (str): Separator for training examples
            example (dict, optional): Context example
            
        Returns:
            tuple: Processed completion and extracted answer
        """
        completion = completion.rstrip().split(train_sep)[0]
        if prompting_style in ["cot", "direct"]:
            return LongContextMCEvaluator.postprocess_qa_style_completion(completion)
        elif prompting_style in ["sat", "satlm"]:
            return LongContextMCEvaluator.postprocess_sat_style_completion(completion)
        else:
            raise RuntimeError("Not implemented")

    @staticmethod
    def postprocess_sat_style_completion(completion):
        """Process completions in SAT-style.
        
        Args:
            completion (str): Raw completion
            
        Returns:
            tuple: Processed completion and extracted answer
        """
        completion = completion.strip()
        print("\n🔄 PROCESSING Z3 CODE FOR EXECUTION")
        print(f"Code length: {len(completion)} characters")
        
        try:
            # True as second parameter enables verbose logging
            _, result = arlsat_satlm_exec(completion, True)
            result = result.strip()
            if "ExecutionError" in result:
                print("❌ EXECUTION ERROR DETECTED")
                result = LongContextMCEvaluator.NULL_ANSWER
            else:
                print(f"✅ SUCCESSFUL EXECUTION - RESULT: {result.lower()}")
                result = result.lower()
        except Exception as e:
            print(f"❌ EXCEPTION DURING Z3 EXECUTION: {str(e)}")
            result = LongContextMCEvaluator.NULL_ANSWER
            
        return completion, result

    @staticmethod
    def postprocess_qa_style_completion(completion):
        """Process completions in question-answering style.
        
        Args:
            completion (str): Raw completion
            
        Returns:
            tuple: Processed completion and extracted answer
        """
        completion = completion.strip()
        if LongContextMCEvaluator.ANSWER_HINT in completion.lower():
            result = completion.lower().strip().split(LongContextMCEvaluator.ANSWER_HINT)[1].strip().rstrip(".")
            # Get the first letter if the format is "A. xxx"
            if len(result) > 1 and result[0] in LongContextMCEvaluator.CHOICES and result[1] == ".":
                result = result[0]
            elif len(result) > 1 and result[0] == "(" and result[1] in LongContextMCEvaluator.CHOICES and result[2] == ")":
                result = result[1]
            # Ensure result is a single letter from the choices
            if len(result) == 1 and result in LongContextMCEvaluator.CHOICES:
                return completion, result
            for choice in LongContextMCEvaluator.CHOICES:
                if result.strip() == choice:
                    return completion, choice
        return completion, LongContextMCEvaluator.NULL_ANSWER

class ArLSATEvaluator(LongContextMCEvaluator):
    """Evaluator specifically for AR-LSAT problems.
    
    Inherits from LongContextMCEvaluator to handle AR-LSAT
    multiple-choice problems.
    """
    pass

class BoardgameQAEvaluator(TaskEvaluator):
    """Evaluator for Boardgame QA problems.
    
    This evaluator handles question-answering tasks about boardgames,
    supporting different prompting styles.
    """
    @staticmethod
    def postprocess_ground_truth(gt):
        """Standardize ground truth for boardgame problems.
        
        Args:
            gt: Raw ground truth
            
        Returns:
            str: Standardized ground truth as yes/no/unknown
        """
        if gt == "proved":
            return "yes"
        elif gt == "disproved":
            return "no"
        elif gt == "unknown":
            return "unknown"
        else:
            raise RuntimeError("Not implemented")

    @classmethod
    def enter_evaluation(cls):
        """Set up random seed for reproducibility in evaluation."""
        random.seed(42)

    @staticmethod
    def postprocess_completion(completion, prompting_style, train_sep, example=None):
        """Process boardgame completions based on prompting style.
        
        Args:
            completion (str): Raw model completion
            prompting_style (str): Style of prompting used
            train_sep (str): Separator for training examples
            example (dict, optional): Context example
            
        Returns:
            tuple: Processed completion and extracted answer
        """
        completion = completion.rstrip().split(train_sep)[0]
        if prompting_style == "cot":
            return BoardgameQAEvaluator.postprocess_cot_style_completion(completion)
        elif prompting_style == "satlm":
            return BoardgameQAEvaluator.postprocess_deafisible_sat_style_completion(completion, prompting_style)
        else:
            raise RuntimeError("Not implemented")

    @staticmethod
    def postprocess_cot_style_completion(completion):
        """Process completions in chain-of-thought style.
        
        Args:
            completion (str): Raw completion
            
        Returns:
            tuple: Processed completion and extracted answer
        """
        if "the answer is" in completion.lower():
            result = completion.lower().strip().split("the answer is")[1].strip().rstrip(".")
        else:
            result = BoardgameQAEvaluator.NULL_ANSWER

        return completion, result

    @staticmethod
    def postprocess_deafisible_sat_style_completion(completion, prompting_style):
        """Process completions in SAT-style for defeasible reasoning.
        
        Args:
            completion (str): Raw completion
            prompting_style (str): Prompting style used
            
        Returns:
            tuple: Processed completion and extracted answer
        """
        completion = completion.strip()
        try:
            _, result = board_satlm_exec(completion, False)
            result = result.strip()
            if "ExecutionError" in result:
                result = BoardgameQAEvaluator.NULL_ANSWER
        except:
            result = BoardgameQAEvaluator.NULL_ANSWER
        return completion, result

    @classmethod
    def generate_random_answer(cls):
        """Generate a random yes/no/unknown answer.
        
        Returns:
            str: Random answer from 'yes', 'no', or 'unknown'
        """
        return random.choice(["yes", "no", "unknown"])

class Boardmaindp1Evaluator(BoardgameQAEvaluator):
    """Evaluator for Boardgame main depth 1 problems."""
    pass

class Boardmaindp2Evaluator(BoardgameQAEvaluator):
    """Evaluator for Boardgame main depth 2 problems."""
    pass

class Boardmaindp3Evaluator(BoardgameQAEvaluator):
    """Evaluator for Boardgame main depth 3 problems."""
    pass

def get_task_evaluator(taskname):
    """Get the appropriate evaluator for a given task.
    
    Args:
        taskname (str): Name of the task
        
    Returns:
        TaskEvaluator: Appropriate evaluator class for the task
    """
    return EVALUATOR_REGISTRY[taskname.lower()]

