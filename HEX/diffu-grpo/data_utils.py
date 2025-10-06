from datasets import load_dataset, Dataset
import pandas as pd
from reward_func import extract_hash_answer

import random
import numpy as np
import torch
import os, sys

def set_random_seed(seed: int = 42):
    # Set the seed for Python's built-in random module
    random.seed(seed)
    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Constants for prompts
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

SUDOKU_SYSTEM_PROMPT = """
Please solve the following 4x4 Sudoku puzzle. The puzzle is provided as a 16-character string reading left-to-right, top-to-bottom, where '0' represents empty cells.

Rules:
- Fill empty cells with digits 1-4
- Each row must contain digits 1-4 exactly once
- Each column must contain digits 1-4 exactly once
- Each 2x2 box must contain digits 1-4 exactly once

Important: Your solution must be a COMPLETE 16-character string with only the digits 1-4, representing your final solved grid.

Respond in this exact format:
<reasoning>
Your step-by-step solving process
</reasoning>
<answer>
[16-character solution string with no spaces or separators]
</answer>
"""


XML_COT_FORMAT = """
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]
    return data.map(
        lambda x: {
            "prompt": [
                {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )


def get_countdown_questions(split="train") -> Dataset:
    data = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split=split)
    data = data.filter(lambda x: len(x["nums"]) == 3)

    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": f"{SYSTEM_PROMPT}\nUsing only the numbers {x['nums']}, create an arithmetic expression that evaluates to exactly {x['target']}. You must use all numbers from the list, and each number must be used exactly once. You may use the operations +, -, *, and / as needed. After reasoning, provide only your final expression inside <answer></answer> tags without including an equals sign or the target number. For example, if the numbers are [2, 3, 4] and the target is 5, a valid answer is: <answer>\n2*4-3\n</answer>",
                },
            ],
            "target": x["target"],
            "numbers": x["nums"],
        }
    )


def get_sudoku_questions() -> Dataset:
    """Load the Sudoku dataset for training or evaluation."""
    cur_path = os.path.dirname(os.path.abspath(__file__))
    sudoku_file_path = "../dataset/4x4_sudoku_unique_puzzles.csv"
    sudoku_file_path = os.path.join(cur_path, sudoku_file_path)
    df = pd.read_csv(sudoku_file_path, dtype={"Puzzle": str, "Solution": str})
    data = Dataset.from_pandas(df)

    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": f"{SUDOKU_SYSTEM_PROMPT}\n\nSolve the following Sudoku puzzle: {x['Puzzle']}\n",
                },
            ],
            "puzzle": x["Puzzle"],
            "solution": x["Solution"],
        }
    )


def get_math_questions(split="train") -> Dataset:
    data = load_dataset("ankner/math-500", split=split)  # type: ignore
    data = data.map(
        lambda x: {  # type: ignore
            "prompt": [
                {
                    "role": "user",
                    "content": f"{SYSTEM_PROMPT}\n\nYou are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{{}}. \n\n{x['problem']}",
                },
            ],
            "answer": x["solution"],
        }
    )  # type: ignore
    return data  # type: ignore
#############################





#############################
# ARCC 
ARC_C_SYSTEM_PROMPT = """You are a science and reasoning expert. You will be given a multiple-choice question with options labeled A, B, C, D (and sometimes E).
Think step by step, but output only ONE final choice.
Wrap ONLY the final choice letter in a \\boxed{}.

Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""
def get_arcc_questions(split="train") -> Dataset:
    def format_question(x):
        stem = str(x.get("question", "")).strip()
        choices = x.get("choices", {})
        texts = choices.get("text", []) or []
        labels = choices.get("label", []) or []
        opts = []
        for lab, txt in zip(labels, texts):
            lab_s = str(lab).strip()
            txt_s = str(txt).strip()
            opts.append(f"{lab_s}. {txt_s}")
        opts_block = "\n".join(opts)
        return f"{stem}\n\nOptions:\n{opts_block}\n\nChoose ONE letter."
    
    data = load_dataset("ai2_arc", "ARC-Challenge", split="train")  # type: ignore
    data = data.map(
        lambda x: {  # type: ignore
            "prompt": [
                {
                    "role": "user",
                    "content": f"{SYSTEM_PROMPT}\n\n{ARC_C_SYSTEM_PROMPT} \n\n{format_question(x)}",
                },
            ],
            "answer": str(x.get("answerKey", "")).strip()
        }
    )  # type: ignore
    return data  # type: ignore
#############################


#############################
# truthfulqa
TRUTHFULQA_SYSTEM_PROMPT = """You are a careful, factual reasoning assistant. You will be given a multiple-choice question with options labeled A, B, C, D.
Think step by step, but output only ONE final choice.
Wrap ONLY the final choice letter in a \\boxed{}.

Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""
def get_truthfulqa_questions() -> Dataset:
    # no existing train dataset for TruthfulQA-MC!!!!!
    def _format_question_with_choices(x) -> str:
        stem = str(x.get("question", "")).strip()
        choices = x.get("choices", []) or []
        labels = [chr(ord("A") + i) for i in range(len(choices))]
        opts = [f"{lab}. {str(txt).strip()}" for lab, txt in zip(labels, choices)]
        opts_block = "\n".join(opts)
        return f"{stem}\n\nOptions:\n{opts_block}\n\nChoose ONE letter."
    
    def _label_to_letter(y, dataset_obj=None) -> str:
        if isinstance(y, str) and len(y) == 1 and y.upper() in "ABCD":
            return y.upper()
        if dataset_obj is not None and "label" in getattr(dataset_obj, "features", {}):
            names = dataset_obj.features["label"].names
            if isinstance(y, int) and 0 <= y < len(names):
                return names[y]
        # fallback
        try:
            return ["A", "B", "C", "D"][int(y)]
        except Exception:
            return "A"  

    # DONT CHANGE THIS !!!!! 
    # IF you want 8:2 split for truthfulQA then 
    # set test_ratio below to 0.2 and same for eval/truthfulqa.py, diffu-grpo/data_utils.py
    set_random_seed(42)
    print(42)

    data = load_dataset(
            "EleutherAI/truthful_qa_mc",
            "multiple_choice",
            split="validation",
        )
    test_ratio = 1 # Default. For eval.
    all_idx = np.arange(len(data))
    test_idx = np.random.choice(len(data), int(len(data) * test_ratio), replace=False)
    train_idx = np.setdiff1d(all_idx, test_idx)
    print('Below indices have to be PERFECTLY SAME as truthfulqa.py def load_test_dataset')
    print("="*30)
    print(test_idx[:20])
    print(train_idx[:20])
    print("="*30)
    print(f'test ratio = {test_ratio}')
    print(f'test idx count = {len(test_idx)}')
    print(f'train idx count = {len(train_idx)}')
    print("="*30)
    truthfulqa_test = data.select(test_idx)
    truthfulqa_train = data.select(train_idx)
    data = truthfulqa_train

    data = data.map(
        lambda x: {  # type: ignore
            "prompt": [
                {
                    "role": "user",
                    "content": f"{SYSTEM_PROMPT}\n\n{TRUTHFULQA_SYSTEM_PROMPT} \n\n{_format_question_with_choices(x)}",
                },
            ],
            "answer": _label_to_letter(x['label'], data)
        }
    )  # type: ignore
    return data  # type: ignore
   
