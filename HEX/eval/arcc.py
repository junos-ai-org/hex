# arc_c.py
import random
import numpy as np
import torch
from datasets import load_dataset
from gsm8k import GSM8KDataset   # 네가 이미 갖고 있는 베이스 클래스
from parsers import Parser, is_equiv

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

class ARCCDataset(GSM8KDataset):
    """
    ARC-C (AI2 Reasoning Challenge - Challenge Set)
    HF dataset: load_dataset("ai2_arc", "ARC-Challenge", split=...)
    Fields:
      - question: str
      - choices: {"text": [..], "label": ["A","B","C","D", ...]}
      - answerKey: str  (e.g., "B")
    NOTE:
      - GSM8K 파서는 숫자 추출 위주이므로, ARC-C는 '선지 문자'를 최종 정답으로 삼는다.
      - 모델이 \boxed{A} 처럼 '문자'만 출력하도록 시스템 프롬프트를 강하게 안내함.
    """

    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=ARC_C_SYSTEM_PROMPT,
        subsample=-1,
        variant="ARC-Challenge",  # "ARC-Challenge" 또는 "ARC-Easy"
    ):
        self.variant = variant
        super().__init__(tokenizer, num_examples, add_reasoning, system_prompt, subsample)

    
    # 평가 대상 split 로드 (기본: test)
    def load_test_dataset(self):
        # ARC는 보통 train/validation/test가 있음. 평가엔 test 사용.
        self.dataset = load_dataset("ai2_arc", self.variant, split="test")

    # few-shot 예시는 같은 데이터셋의 train split에서 뽑아와서
    # '선지 포함 질문' + '정답 라벨' 형식으로 넣어준다.
    def load_few_shot_examples(self):
        train = load_dataset("ai2_arc", self.variant, split="train")
        few_shot_examples = []
        if self.num_examples > 0:
            idxs = random.sample(range(len(train)), min(self.num_examples, len(train)))
            for i in idxs:
                row = train[i]
                q = self._format_question_with_choices(row)
                a = str(row.get("answerKey", "")).strip()
                few_shot_examples.append({"question": q, "answer": a})
        return few_shot_examples

    # ARC row -> "문제 + 선택지" 문자열 생성
    def _format_question_with_choices(self, row: dict) -> str:
        stem = str(row.get("question", "")).strip()
        choices = row.get("choices", {})
        texts = choices.get("text", []) or []
        labels = choices.get("label", []) or []
        # 라벨과 텍스트를 묶어서 보기 좋게 정렬
        opts = []
        for lab, txt in zip(labels, texts):
            lab_s = str(lab).strip()
            txt_s = str(txt).strip()
            opts.append(f"{lab_s}. {txt_s}")
        opts_block = "\n".join(opts)
        # 최종 프롬프트용 문장
        return f"{stem}\n\nOptions:\n{opts_block}\n\nChoose ONE letter."

    def __getitem__(self, idx):
        row = self.dataset[self.subsample[idx].item()]
        question = self._format_question_with_choices(row)
        # ARC의 정답은 문자 라벨 (예: "A","B")
        answer = str(row.get("answerKey", "")).strip()
        prompt = self.create_prompt(question)
        return prompt, question, answer
