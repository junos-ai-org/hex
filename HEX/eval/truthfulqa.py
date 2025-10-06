# truthfulqa.py
import random
import numpy as np
import torch
from datasets import load_dataset
from gsm8k import GSM8KDataset
from parsers import Parser, is_equiv  

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

class TruthfulQAMCDataset(GSM8KDataset):
    """
    TruthfulQA (multiple-choice, simplified: 4 options)
    HF: load_dataset("EleutherAI/truthful_qa_mc", "multiple_choice", split="validation")

    Fields per row:
      - question: str
      - choices: List[str]  (length 4)
      - label: int (0..3)  # ClassLabel(names=["A","B","C","D"])
    """

    def __init__(
        self,
        tokenizer,
        num_examples: int = 0,
        add_reasoning: bool = True,
        system_prompt: str = TRUTHFULQA_SYSTEM_PROMPT,
        subsample: int = -1,
    ):
        super().__init__(tokenizer, num_examples, add_reasoning, system_prompt, subsample)

    def load_test_dataset(self):
        self.dataset = load_dataset(
            "EleutherAI/truthful_qa_mc",
            "multiple_choice",
            split="validation",
        )
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
        


        # DONT CHANGE THIS !!!!! 
        # IF you want 8:2 split for truthfulQA then 
        # set test_ratio below to 0.2 and same for eval/truthfulqa.py, diffu-grpo/data_utils.py
        set_random_seed(42)
        print(42)
      

        test_ratio = 1 # Default. For eval.
        all_idx = np.arange(len(self.dataset))
        test_idx = np.random.choice(len(self.dataset), int(len(self.dataset) * test_ratio), replace=False)
        train_idx = np.setdiff1d(all_idx, test_idx)
        print('Below indices have to be PERFECTLY SAME as data_utils.py def get_truthfulqa_questions')
        print("="*30)
        print(test_idx[:20])
        print(train_idx[:20])
        print("="*30)
        print(f'test ratio = {test_ratio}')
        print(f'test idx count = {len(test_idx)}')
        print(f'train idx count = {len(train_idx)}')
        print("="*30)
        truthfulqa_test = self.dataset.select(test_idx)
        truthfulqa_train = self.dataset.select(train_idx)
        self.dataset = truthfulqa_test

 
    def load_few_shot_examples(self):
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
        

        src = load_dataset(
            "EleutherAI/truthful_qa_mc",
            "multiple_choice",
            split="validation",
        )
        few_shot_examples = []
        if self.num_examples > 0:
            idxs = random.sample(range(len(src)), min(self.num_examples, len(src)))
            for i in idxs:
                row = src[i]
                q = self._format_question_with_choices(row)
                a = self._label_to_letter(row["label"], src)  # 'A'..'D'
                few_shot_examples.append({"question": q, "answer": a})
        return few_shot_examples

    def _format_question_with_choices(self, row: dict) -> str:
        stem = str(row.get("question", "")).strip()
        choices = row.get("choices", []) or []
  
        labels = [chr(ord("A") + i) for i in range(len(choices))]
        opts = [f"{lab}. {str(txt).strip()}" for lab, txt in zip(labels, choices)]
        opts_block = "\n".join(opts)
        return f"{stem}\n\nOptions:\n{opts_block}\n\nChoose ONE letter."

    def _label_to_letter(self, y, dataset_obj=None) -> str:
   
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
            return "A"  # 안전 기본값

    def __getitem__(self, idx):
        row = self.dataset[self.subsample[idx].item()]
        question = self._format_question_with_choices(row)
        answer = self._label_to_letter(row["label"], self.dataset)  # 'A'..'D'
        prompt = self.create_prompt(question)
        return prompt, question, answer
