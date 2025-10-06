# save as make_team_image.py
from PIL import Image
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Patch
import time
from tqdm import tqdm
from parser_helper import is_equiv, remove_boxed, last_boxed_only_string
from collections import Counter, defaultdict

# def get_all_file_paths(folder_path, dataset_type, exclude_list = [256]): #error.
#         file_paths = []
#         for root, dirs, files in os.walk(folder_path):
#             for file in files:
#                 if dataset_type in file:
#                     if exclude_list :
#                         for num in exclude_list:
#                             if f'block{num}' not in os.path.join(root, file):
#                                 file_paths.append(os.path.join(root, file))
#                     else:
#                         file_paths.append(os.path.join(root, file))
#         return file_paths

# def get_all_file_paths_2(folder_path, dataset_type, include_str_list = [], exclude_str_list = []):
#         file_paths = []
#         for root, dirs, files in os.walk(folder_path):
#             for file in files:
#                 if dataset_type in file:
#                     if include_str_list :
#                         for string in include_str_list:
#                             if string in os.path.join(root, file):
#                                 file_paths.append(os.path.join(root, file))
#                     else:
#                         file_paths.append(os.path.join(root, file))
#         return file_paths

# import os

def get_all_file_paths_2(folder_path, dataset_type,
                         include_str_list=None, exclude_str_list=None):
    if include_str_list is None:
        include_str_list = []
    if exclude_str_list is None:
        exclude_str_list = []

    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if dataset_type in file:
                full_path = os.path.join(root, file)

                # 🔹 exclude 문자열이 하나라도 포함되어 있으면 skip
                if exclude_str_list and any(ex in full_path for ex in exclude_str_list):
                    continue

                if include_str_list:
                    if any(inc in full_path for inc in include_str_list):
                        file_paths.append(full_path)
                
                else:
                    file_paths.append(full_path)

    return file_paths




def gsm_extract_ans_from_generation(out_nl):
    import re
    parsed_answer = None
    boxed_matches = re.findall(r"\\boxed{(.*?)}", out_nl)
    if boxed_matches:
        for boxed_content in boxed_matches:
            boxed_content = boxed_content.strip()
            if boxed_content and boxed_content != "..." and not re.match(r"^\.+$", boxed_content):
                try:
                    parsed_answer = float(boxed_content)
                    break
                except ValueError:
                    numbers = re.findall(r"-?\d+\.?\d*", boxed_content)
                    if numbers:
                        try:
                            parsed_answer = float(numbers[0])
                            break
                        except ValueError:
                            pass
    if parsed_answer is None:
        answer_match = re.search(r"<answer>(.*?)</answer>", out_nl, re.DOTALL)
        if answer_match:
            answer_text = answer_match.group(1).strip()
            if answer_text:
                try:
                    parsed_answer = float(answer_text)
                except ValueError:
                    numbers = re.findall(r"-?\d+\.?\d*", answer_text)
                    if numbers:
                        try:
                            parsed_answer = float(numbers[-1])
                        except ValueError:
                            pass
    return parsed_answer

def parse_math_answers(raw_generation):
    parsed_answer = None
    try:
        parsed_answer = remove_boxed(last_boxed_only_string(raw_generation))
    except:
        parsed_answer = None

    if not parsed_answer:
        answer_match = re.search(r"<answer>(.*?)</answer>", raw_generation, re.DOTALL)
        if answer_match:
            parsed_answer = answer_match.group(1).strip()
    return parsed_answer

def parse_countdown_answers(output_nl_each_strategy):
    
    def validate_equation(equation_str, available_numbers):
        """Validate that equation only uses available numbers and each number once."""
        try:
            numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
            available_numbers = sorted(available_numbers)
            numbers_in_eq = sorted(numbers_in_eq)
            return numbers_in_eq == available_numbers
        except:
            return False

    def evaluate_equation(equation_str):
        """Safely evaluate the arithmetic equation."""
        try:
            allowed_pattern = r"^[\d+\-*/().\s]+$"
            if not re.match(allowed_pattern, equation_str):
                raise ValueError("Invalid characters in equation.")
            result = eval(equation_str.strip(), {"__builtins__": None}, {})
            return result
        except Exception:
            return float("Inf")

    
    generated_text = output_nl_each_strategy
    equation = ""
    try:
        equation = remove_boxed(last_boxed_only_string(generated_text))
    except:
        # Try to extract from answer tags
        answer_match = re.search(r"<answer>(.*?)</answer>", generated_text, re.DOTALL)
        if answer_match:
            equation = answer_match.group(1).strip()
        else:
            equation = generated_text

    # Replace LaTeX operators with Python operators
    equation = equation.replace(r"\div", "/").replace(r"\times", "*").replace(r"\cdot", "*")
    return equation

def evaluate_countdown(voted_answer, ground_truth):
    
    def validate_equation(equation_str, available_numbers):
        """Validate that equation only uses available numbers and each number once."""
        try:
            numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
            available_numbers = sorted(available_numbers)
            numbers_in_eq = sorted(numbers_in_eq)
            return numbers_in_eq == available_numbers
        except:
            return False

    def evaluate_equation(equation_str):
        """Safely evaluate the arithmetic equation."""
        try:
            allowed_pattern = r"^[\d+\-*/().\s]+$"
            if not re.match(allowed_pattern, equation_str):
                raise ValueError("Invalid characters in equation.")
            result = eval(equation_str.strip(), {"__builtins__": None}, {})
            return result
        except Exception:
            return float("Inf")
    # Extract available numbers and target from ground_truth
    numbers = []
    target = None

    if isinstance(ground_truth, list) and len(ground_truth) == 2:
        numbers = ground_truth[0]
        target = ground_truth[1]
    else:
        # Fallback to parsing from question if ground_truth is not in expected format
        numbers_match = re.search(r"Numbers: \[([\d, ]+)\]", question, re.IGNORECASE)
        if numbers_match:
            numbers_str = numbers_match.group(1)
            numbers = [int(num.strip()) for num in numbers_str.split(",")]

        target_match = re.search(r"Target: (\d+)", question, re.IGNORECASE)
        if target_match:
            target = int(target_match.group(1))
    ######################
    # Check for equation with equals sign and extract only the expression part
    equation = voted_answer
    equation_match = re.search(r"([0-9+\-*/() ]+)=[0-9. ]+", equation)
    if equation_match:
        equation = equation_match.group(1).strip()

    is_correct = False
    result = None

    # Validate and evaluate the equation
    is_valid = validate_equation(equation, numbers)
    if is_valid:
        result = evaluate_equation(equation)
        if target is not None and abs(result - target) < 1e-5:
            is_correct = True
            
    return is_correct


def parse_sudoku_answers(question, output_nl):

    question = question
    raw_generation = output_nl

    # Extract puzzle
    puzzle_str = ""
    if len(question) >= 16 and all(c.isdigit() or c == "0" for c in question[:16]):
        puzzle_str = question[:16]
    else:
        match = re.search(r"Sudoku puzzle: ([0-9]{16})", question)
        if match:
            puzzle_str = match.group(1)
    assert len(puzzle_str) == 16, f"Invalid puzzle string: {puzzle_str}"

    empty_indices = [i for i in range(16) if puzzle_str[i] == "0"]
    empty_cells = len(empty_indices)

    # Extract solution using regex patterns
    solution_str = ""
    patterns = [
        r"<answer>.*?```\s*([\d\s]+)```",
        r"<answer>(.*?)(?:<\|eot_id\|>|<\|endoftext\|>|</answer>)",
        r"</answer>\s*(.*?)(?:<\|eot_id\|>|<\|endoftext\|>|$)",
        r".*?(\d{16})\s*</answer>",
        r"\b(\d{16})\b",
    ]

    for pattern in patterns:
        if solution_str:
            break
        match = re.search(pattern, raw_generation, re.DOTALL)
        if match and match.group(1).strip():
            solution_str = match.group(1).strip()

    solution_str = re.sub(r"\s", "", solution_str)
    return solution_str

def evaluate_sudoku_answers(extracted_answer, question, ground_truth):

    # Extract puzzle
    puzzle_str = ""
    if len(question) >= 16 and all(c.isdigit() or c == "0" for c in question[:16]):
        puzzle_str = question[:16]
    else:
        match = re.search(r"Sudoku puzzle: ([0-9]{16})", question)
        if match:
            puzzle_str = match.group(1)
    assert len(puzzle_str) == 16, f"Invalid puzzle string: {puzzle_str}"

    empty_indices = [i for i in range(16) if puzzle_str[i] == "0"]
    empty_cells = len(empty_indices)
    
    solution_str = extracted_answer

    # Handle solution length
    if not solution_str:
        correct_cells = 0
    else:
        if len(solution_str) < 16:
            solution_str = solution_str + "0" * (16 - len(solution_str))
        elif len(solution_str) > 16:
            solution_str = solution_str[:16]
        correct_cells = sum(1 for i in empty_indices if solution_str[i] == ground_truth[i])

    accuracy = correct_cells / empty_cells if empty_cells > 0 else 0.0
    return correct_cells, empty_cells # A,B  매 생성마다, A,B를 각각 누적시키고 비율을 최종적으로 구해야함 
    


def parse_arcc_answers(json_path=None, json_data=None):
    """
    Parse ARC-c style multiple-choice answers from generations.
    Extracts a single choice among {'A','B','C','D'} from model output.

    Expects input JSON with a top-level key "generations", where each item may have:
      - "question": str
      - "ground_truth": str  # expected to be 'A'/'B'/'C'/'D' (case-insensitive)
      - "generations": str   # model's raw output

    Relies on external helpers if available:
      - count_effective_tokens(text)
      - is_equiv(pred, gold)  # optional; if unavailable, falls back to strict equality

    Returns:
      (total_correct, total_processed, processed_items, total_effective_tokens)
    """
    if json_path:
        with open(json_path, "r") as f:
            data = json.load(f)
    else:
        data = json_data

    total_correct = 0
    total_processed = 0
    total_effective_tokens = 0
    processed_items = []

    def norm_choice(s):
        if not s:
            return None
        s = s.strip().upper()
        # Strip common trailing punctuation
        s = re.sub(r'[)\].,;:>\s]+$', '', s)
        # Strip leading punctuation
        s = re.sub(r'^[<\[(\s*]+', '', s)
        return s if s in {"A", "B", "C", "D"} else None

    def last_match(groups):
        """Return the last non-empty normalized ABCD from a list of regex matches (tuples or strings)."""
        candidate = None
        for g in groups:
            if isinstance(g, tuple):
                for part in g[::-1]:
                    cand = norm_choice(part)
                    if cand:
                        candidate = cand
                        break
            else:
                cand = norm_choice(g)
                if cand:
                    candidate = cand
        return candidate

    def extract_choice_letter(text):
        """
        Try multiple increasingly permissive patterns; prefer the LAST occurrence found.
        """
        if not text:
            return None

        # 1) Explicit XML-ish tag
        m = list(re.finditer(r"<answer>\s*([ABCD])\s*</answer>", text, flags=re.IGNORECASE|re.DOTALL))
        if m:
            return norm_choice(m[-1].group(1))

        # 2) Boxed answers like \boxed{C} or \fbox{B}
        m = list(re.finditer(r"\\(?:boxed|fbox)\s*\{\s*([ABCD])\s*\}", text, flags=re.IGNORECASE))
        if m:
            return norm_choice(m[-1].group(1))

        # # 3) Phrases: "Answer: C", "Correct answer is B", "정답은 C", "답: D"
        # m = list(re.finditer(
        #     r"(?:(?:final|correct)?\s*answer|ans\.?|정답|답)\s*[:\-–]?\s*([ABCD])\b",
        #     text, flags=re.IGNORECASE))
        # if m:
        #     return norm_choice(m[-1].group(1))

        # 4) Words: option/choice (영문)
        m = list(re.finditer(r"(?:option|choice)\s*[:\-–]?\s*([ABCD])\b", text, flags=re.IGNORECASE))
        if m:
            return norm_choice(m[-1].group(1))

        # 5) Parenthesized or bracketed forms like (A), [B], <C>, **D**
        m = list(re.finditer(r"[\(\[\<]\s*([ABCD])\s*[\)\]\>]", text, flags=re.IGNORECASE))
        if m:
            return norm_choice(m[-1].group(1))

        # # 6) Standalone letter at end of line/sentence with light punctuation, preceded by cue words
        # m = list(re.finditer(
        #     r"(?:is|thus|therefore|so|결론적으로|따라서|즉)\s+([ABCD])\s*[.)]?(?:\s|$)",
        #     text, flags=re.IGNORECASE))
        # if m:
        #     return norm_choice(m[-1].group(1))

        # 7) Bold/markdown emphasis like **A** or *B*
        m = list(re.finditer(r"\*\*\s*([ABCD])\s*\*\*|\*\s*([ABCD])\s*\*", text, flags=re.IGNORECASE))
        if m:
            # find any capturing group with content
            for match in m[::-1]:
                g = match.group(1) or match.group(2)
                cand = norm_choice(g)
                if cand:
                    return cand

        # 8) Very permissive: last solitary A-D token on its own line or before EOL
        # (avoid swallowing letters inside words)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for ln in reversed(lines):
            # prioritize patterns like "=> C", "= C", "-> C"
            m = re.search(r"(?:=>|->|=|:)\s*([ABCD])\s*$", ln, flags=re.IGNORECASE)
            if m:
                cand = norm_choice(m.group(1)); 
                if cand: 
                    return cand
            # then a bare token at line end
            m = re.search(r"\b([ABCD])\b[.)]?\s*$", ln, flags=re.IGNORECASE)
            if m:
                cand = norm_choice(m.group(1))
                if cand:
                    return cand

        return None

        # Note: if you already have helpers like last_boxed_only_string/remove_boxed,
        # you can optionally call them before/after the patterns above.

    for item in data.get("generations", []):
        total_processed += 1
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        raw_generation = item.get("generations", "")

        # Count effective tokens (external helper assumed to exist)
        try:
            effective_tokens = count_effective_tokens(raw_generation)
        except NameError:
            # Fallback if helper is unavailable
            effective_tokens = len(raw_generation.split())
        total_effective_tokens += effective_tokens

        # Try extraction
        parsed_answer = extract_choice_letter(raw_generation)

        # Also allow <answer>...</answer> fallback if it contained full text before
        if parsed_answer is None:
            m = re.search(r"<answer>(.*?)</answer>", raw_generation, re.DOTALL|re.IGNORECASE)
            if m:
                parsed_answer = norm_choice(m.group(1))

        # Compare with ground truth
        is_correct = False
        if parsed_answer is not None and ground_truth:
            gt = norm_choice(ground_truth) or ground_truth.strip().upper()
            try:
                is_correct = is_equiv(parsed_answer, gt)  # external helper (if provided)
            except NameError:
                is_correct = (parsed_answer == gt)

        if is_correct:
            total_correct += 1

        processed_items.append(
            {
                "question": question,
                "raw_generation": raw_generation,
                "extracted_answer": parsed_answer,  # 'A'/'B'/'C'/'D' or None
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "effective_tokens": effective_tokens,
            }
        )

    return (
        total_correct,
        total_processed,
        processed_items,
        total_effective_tokens,
    )





################################
from collections import Counter
def most_frequent_element(data):
    counter = Counter(data)
    return counter.most_common(1)[0][0]

def most_frequent_element_2(data):
    counter = Counter(data)
    most_common = counter.most_common()
    top_elem, top_count = most_common[0]

    # 최빈값이 여러 개인지 확인
    is_tied = sum(1 for _, count in most_common if count == top_count) > 1
    
    return top_elem, is_tied

import random
from collections import Counter
import itertools
def most_frequent_element_3(data):
    counter = Counter(data)
    most_common = counter.most_common()
    top_count = most_common[0][1]

    # 최빈값 후보들만 모으기
    candidates = [elem for elem, count in most_common if count == top_count]

    # 동률 여부
    is_tied = len(candidates) > 1

    # 동률이면 랜덤 선택, 아니면 그대로 반환
    chosen = random.choice(candidates)

    return chosen, is_tied
def most_frequent_element_4(data):
    counter = Counter(data)
    most_common = counter.most_common()
    top_count = most_common[0][1]

    # 최빈값 후보들만 모으기
    candidates = [elem for elem, count in most_common if count == top_count]

    # 동률 여부
    is_tied = len(candidates) > 1

    # 동률이면 랜덤 선택, 아니면 그대로 반환
    chosen = random.choice(candidates)

    avg_distance = None
    if not is_tied and top_count > 1:
        # chosen이 등장한 모든 위치 인덱스
        positions = [i for i, val in enumerate(data) if val == chosen]
        
        # 모든 쌍의 거리 계산
        distances = [abs(i - j) for i, j in itertools.combinations(positions, 2)]
        
        # 평균 거리
        avg_distance = sum(distances) / len(distances)

    return chosen, is_tied, avg_distance

import statistics


def most_frequent_elements(data):
    counter = Counter(data)
    max_count = max(counter.values())  # 가장 큰 빈도
    most_common = counter.most_common()
    top_count = most_common[0][1]
    is_tied = sum(1 for _, count in most_common if count == top_count) > 1
    return [k for k, v in counter.items() if v == max_count], is_tied

def most_frequent_elements_exclude_None(data):
    data = [(i, -ll, -ll_exclude_AEoT) for i, ll, ll_exclude_AEoT in data if i is not None]

    counter = Counter(data)
    max_count = max(counter.values())  # 가장 큰 빈도
    most_common = counter.most_common()
    top_count = most_common[0][1]
    is_tied = sum(1 for _, count in most_common if count == top_count) > 1
    return [k for k, v in counter.items() if v == max_count], is_tied

def most_low_nll_elements_exclude_None(data, use_ll = 'll'):
    # print(data)
    data = [(i, -ll, -ll_exclude_AEoT) for i, ll, ll_exclude_AEoT in data if i is not None]
    if use_ll == 'll':
        min_ll = min(ll for _, ll, _ in data)
        best_indices = [i for i, ll, _ in data if ll == min_ll]
    elif use_ll == 'll_notAEoT':
        min_ll = min(ll_notEoT for _, _, ll_notEoT in data)
        best_indices = [i for i, _, ll in data if ll == min_ll]
    
    is_tied = len(best_indices) > 1
    return best_indices, is_tied


# def most_frequent_elements_exclude_None_and_tie_lownll(data, selecting_rule='', exclude_None = True):
#     # None 제거 및 -ll, -ll_exclude_AEoT 변환
    
#     if exclude_None:
#         data = [(i, -ll) for i, ll in data if i is not None]
#     else:
#         data = [(i, -ll) for i, ll in data]

#     if data == []:
#         data = [(None,0)]

#     # data = [(i, -ll, -ll_exclude_AEoT) for i, ll, ll_exclude_AEoT in data]
#     if selecting_rule == 'frequency':
#         # 빈도 세기
#         counter = Counter([i for i, _ in data])
#         max_count = max(counter.values())
#         most_common = counter.most_common()
#         top_count = most_common[0][1]
#         is_tied = sum(1 for _, count in most_common if count == top_count) > 1
#         return [k for k, v in counter.items() if v == max_count], is_tied

#     elif selecting_rule == 'lowest_nll':
#         # 빈도 세기
#         counter = Counter([i for i, _ in data])
#         max_count = max(counter.values())
#         most_common = counter.most_common()
#         top_count = most_common[0][1]
#         is_tied = sum(1 for _, count in most_common if count == top_count) > 1

#         min_ll = min(nll for _, nll in data)
#         best_indices = [i for i, nll in data if nll == min_ll]
#         return best_indices, is_tied

#     elif selecting_rule == 'frequency_tie_lowest_nll':
#         # 빈도 세기
#         counter = Counter([i for i, _ in data])
#         max_count = max(counter.values())
#         most_common = counter.most_common()
#         top_count = most_common[0][1]
#         is_tied = sum(1 for _, count in most_common if count == top_count) > 1
#         # i별로 ll 값 저장
#         ll_map = defaultdict(list)
#         for i, nll in data:
#             ll_map[i].append(nll)  # 필요하면 ll_exclude_AEoT도 같이 저장 가능
#         # 최빈값 i와 그에 대응하는 ll 값들 추출
#         # result = [(k, ll_map[k]) for k, v in counter.items() if v == max_count]
#         result = [(k, min(ll_map[k])) for k, v in counter.items() if v == max_count]
#         result.sort(key=lambda x: x[1], reverse=False)
#         return [result[0][0]], is_tied


from collections import Counter, defaultdict

def most_frequent_elements_exclude_None_and_tie_lownll(
    data, selecting_rule='', exclude_None=True
):
    """
    data: list of tuples (i, ll) where 'll' is log-likelihood (will be negated to nll)
    Returns:
        (list_of_(value, nll), is_tied_among_frequencies)
    """
    # None 제거 및 -ll -> nll로 변환
    if exclude_None:
        data = [(i, -ll) for i, ll in data if i is not None]
    else:
        data = [(i, -ll) for i, ll in data]

    if not data:
        return [(None, 0)], False

    # 공통 준비물: 빈도 계산, 동률 여부
    counter = Counter([i for i, _ in data])
    max_count = max(counter.values())
    most_common = counter.most_common()
    top_count = most_common[0][1]
    is_tied = sum(1 for _, count in most_common if count == top_count) > 1

    # 값별 nll 리스트/요약
    ll_map = defaultdict(list)
    for i, nll in data:
        ll_map[i].append(nll)

    if selecting_rule == 'frequency':
        # 최빈값들의 (값, 최소 nll) 모두 반환
        result = [(k, min(ll_map[k])) for k, v in counter.items() if v == max_count]
        return result, is_tied

    elif selecting_rule == 'lowest_nll':
        # 전역 최소 nll을 갖는 값(들) 반환 (중복 제거)
        min_ll = min(nll for _, nll in data)
        best_values = {i for i, nll in data if nll == min_ll}
        result = [(i, min_ll) for i in best_values]
        return result, is_tied

    elif selecting_rule == 'frequency_tie_lowest_nll':
        # 최빈값 집합에서 nll이 가장 작은 값(들)만 반환
        top_items = [(k, min(ll_map[k])) for k, v in counter.items() if v == max_count]
        if not top_items:
            return [(None, 0)], is_tied
        min_top_nll = min(nll for _, nll in top_items)
        result = [(k, nll) for k, nll in top_items if nll == min_top_nll]
        return result, is_tied

    else:
        # 규칙 미지정: 안전하게 전역 최소 nll 반환
        min_ll = min(nll for _, nll in data)
        best_values = {i for i, nll in data if nll == min_ll}
        result = [(i, min_ll) for i in best_values]
        return result, is_tied










# a = [1,1,2,2,3]
# print(most_frequent_elements(a))  # [1, 2]
# exit()
def mean_std_pm(data):
    # 평균
    mean_val = statistics.mean(data)
    # 표본 표준편차 (n-1로 나눔)
    std_val = statistics.stdev(data)  
    
    # 보기 좋게 포맷팅 (소수점 3자리)
    return f"{mean_val:.2f} ± {std_val:.2f}"

def common_prefix_length(strings):
    if not strings:
        return 0

    min_len = min(len(s) for s in strings)  # 가장 짧은 문자열 길이까지만 비교
    prefix_len = 0

    for i in range(min_len):
        # i번째 위치의 모든 문자열 문자가 같은지 확인
        chars = {s[i] for s in strings}
        if len(chars) == 1:  # 모두 동일
            prefix_len += 1
        else:
            break

    return prefix_len


import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def is_missing(x):
    # None 또는 NaN 모두 검출
    return x is None or (isinstance(x, float) and math.isnan(x))

def make_answer_grid_from_tuples(
    extracted_list_for_all_datapoints,   # (N, 6) with (pred, score)
    majvoted_list_for_all_datapoints,    # (N, 1) -> [pred]
    gt_list_for_all_datapoints,          # (N,) -> pred or None
    xlabels=("8","16","32","64","128","256","majvote","gt"),
    savepath=None,
    show=True,
    start=0,
    n=None,
    indices=None
):
    N_total = len(extracted_list_for_all_datapoints)
    assert len(majvoted_list_for_all_datapoints) == N_total
    assert len(gt_list_for_all_datapoints) == N_total
    assert all(len(row) == 6 for row in extracted_list_for_all_datapoints)

    # --- 서브셋 선택 ---
    if indices is not None:
        sel = list(indices)
    else:
        if n is None:
            end = N_total
        else:
            end = min(N_total, start + n)
        sel = list(range(start, end))

    # 🔎 디버그 출력: sel 크기와 앞 20개
    print("DEBUG sel:", len(sel), sel[:20])

    if len(sel) == 0:
        raise ValueError("선택된 행이 없습니다. start/n 또는 indices를 확인하세요.")

    # 헬퍼
    def get_pred(x):
        if isinstance(x, (tuple, list)) and len(x) >= 1:
            return x[0]
        return x

    # 선택된 부분만 추출
    ext = [extracted_list_for_all_datapoints[i] for i in sel]
    maj = [row[0] if isinstance(row, (list, tuple)) and len(row) > 0 else None
        for row in (majvoted_list_for_all_datapoints[i] for i in sel)]
    gt  = [gt_list_for_all_datapoints[i] for i in sel]

    N = len(sel)
    grid_rgb = np.ones((N, 8, 3), dtype=float)

    white_grid = [1, 1, 1]
    black_grid = [0, 0, 0]
    blue_grid = [0, 0, 1]

    # None → 검정, GT열 기본 파랑
    for i in range(N):
        for j in range(6):
            pred = get_pred(ext[i][j])
            if pred is None:
                grid_rgb[i, j] = white_grid
        if maj[i] is None:
            grid_rgb[i, 6] = white_grid
        if gt[i] is None:
            grid_rgb[i, 7] = white_grid
        else:
            grid_rgb[i, 7] = blue_grid

    # GT와 비교
    for i in range(N):
        g = gt[i]
        if g is None:
            continue
        for j in range(6):
            pred = get_pred(ext[i][j])
            if pred is None:
                continue
            grid_rgb[i, j] = blue_grid if pred == g else white_grid
        if maj[i] is not None:
            grid_rgb[i, 6] = blue_grid if maj[i] == g else white_grid

    # 플롯
    fig_h = max(4, min(16, int(N/40)))
    fig, ax = plt.subplots(figsize=(8, fig_h), dpi=120)
    ax.imshow(grid_rgb, aspect='auto', interpolation='nearest')
    ax.set_xticks(range(8))
    ax.set_xticklabels(xlabels)
    ax.set_yticks([])
    ax.set_xlabel("Methods")
    ax.set_title(f"Answers (rows={N} / total={N_total}) — blue=match GT, white=mismatch, black=None")

    ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.3)
    ax.tick_params(which='minor', bottom=False, left=False)

    legend_patches = [
        Patch(facecolor='black', edgecolor='black', label='None'),
        Patch(facecolor='white', edgecolor='black', label='Mismatch'),
        Patch(facecolor='blue', edgecolor='black', label='Match (and GT)'),
    ]
    ax.legend(handles=legend_patches, loc='upper right', frameon=True)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def make_answer_grid_from_tuples_2(
    extracted_list_for_all_datapoints,   # (N, 6) with (pred, score)
    majvoted_list_for_all_datapoints,    # (N, 1) -> [pred]
    gt_list_for_all_datapoints,          # (N,) -> pred or None
    xlabels=("8","16","32","64","128","256","majvote","gt"),
    savepath=None,
    show=True,
    start=0,
    n=None,
    indices=None
):
    N_total = len(extracted_list_for_all_datapoints)
    assert len(majvoted_list_for_all_datapoints) == N_total
    assert len(gt_list_for_all_datapoints) == N_total
    assert all(len(row) == 6 for row in extracted_list_for_all_datapoints)

    # --- subset 선택 ---
    if indices is not None:
        sel = list(indices)
    else:
        end = N_total if n is None else min(N_total, start + n)
        sel = list(range(start, end))
    print("DEBUG sel:", len(sel), sel[:20])
    if not sel:
        raise ValueError("선택된 행이 없습니다. start/n 또는 indices를 확인하세요.")

    # 헬퍼
    def get_pred(x):
        if isinstance(x, (tuple, list)) and len(x) >= 1:
            return x[0]
        return x

    # 선택된 부분만 추출
    ext = [extracted_list_for_all_datapoints[i] for i in sel]
    maj = [row[0] if isinstance(row, (list, tuple)) and len(row) > 0 else None
           for row in (majvoted_list_for_all_datapoints[i] for i in sel)]
    gt  = [gt_list_for_all_datapoints[i] for i in sel]

    N = len(sel)
    grid_rgb = np.ones((N, 8, 3), dtype=float)

    WHITE = [1, 1, 1]
    BLACK = [0, 0, 0]
    BLUE  = [0, 0, 1]

    # None → 검정, GT열 기본 파랑
    for i in range(N):
        for j in range(6):
            pred = get_pred(ext[i][j])
            if pred is None:
                grid_rgb[i, j] = WHITE
        if maj[i] is None:
            grid_rgb[i, 6] = WHITE
        if gt[i] is None:
            grid_rgb[i, 7] = WHITE
        else:
            grid_rgb[i, 7] = BLUE

    # GT와 비교
    for i in range(N):
        g = gt[i]
        if g is None:
            continue
        for j in range(6):
            pred = get_pred(ext[i][j])
            if pred is None:
                continue
            grid_rgb[i, j] = BLUE if pred == g else WHITE
        if maj[i] is not None:
            grid_rgb[i, 6] = BLUE if maj[i] == g else WHITE

    # --- 열별 정확도 계산 (GT != None인 행 기준) ---
    valid_rows = [i for i in range(N) if gt[i]]
    denom = len(valid_rows) 

    acc = []
    # 6개 extracted
    for j in range(6):
        correct = 0
        for i in valid_rows:
            pred = get_pred(ext[i][j])
            if pred == gt[i]:
                correct += 1
        acc.append(100.0 * correct / denom)
    # majvote
    correct = 0
    for i in valid_rows:
        if maj[i] == gt[i] and gt[i] is not None:
            correct += 1
    acc.append(100.0 * correct / denom)
    # gt열은 비교 대상이 아니므로 표시만
    acc.append(np.nan)

    # --- 플롯: 그리드 + 아래에 퍼센트 표기 ---
    # 아래에 숫자 라벨을 넣을 영역을 따로 만들어 보기가 좋아요.
    dpi = 120
    fig = plt.figure(figsize=(8, max(4, min(16, int(N/40)))), dpi=dpi)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[10, 1], hspace=0.05)

    ax = fig.add_subplot(gs[0])
    ax.imshow(grid_rgb, aspect='auto', interpolation='nearest')
    ax.set_xticks(range(8))
    ax.set_xticklabels(xlabels)
    ax.set_yticks([])
    ax.set_xlabel("Methods")
    ax.set_title(
        f"Answers (rows={N} / total={N_total}) — blue=match GT, white=mismatch, black=None"
    )

    # 얇은 그리드
    ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.3)
    ax.tick_params(which='minor', bottom=False, left=False)

    # 아래 퍼센트 영역
    axb = fig.add_subplot(gs[1], sharex=ax)
    axb.set_xlim(-0.5, 7.5)
    axb.set_ylim(0, 1)
    axb.axis('off')

    # 각 열 중앙에 퍼센트 텍스트 (GT 열은 '—')
    for j in range(8):
        label = "—" if j == 7 or np.isnan(acc[j]) else f"{acc[j]:.2f}%"
        axb.text(j, 0.5, label, ha='center', va='center', fontsize=10)

    # 범례
    legend_patches = [
        Patch(facecolor='black', edgecolor='black', label='None'),
        Patch(facecolor='white', edgecolor='black', label='Mismatch'),
        Patch(facecolor='blue',  edgecolor='black', label='Match (and GT)'),
    ]
    ax.legend(handles=legend_patches, loc='upper right', frameon=True)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)

def build_correct_blocks_table(
    extracted_list_for_all_datapoints,   # (N, 6) with items like (pred, score)
    gt_list_for_all_datapoints,          # (N,) -> pred or None
    block_labels=("8","16","32","64","128","256"),
    start=0, n=None, indices=None,
    comparator=None,                     # 예: math면 lambda p,g: is_equiv(p,g)
    none_token="None"                    # None을 어떻게 표기할지
):
    # 기본 비교: gsm처럼 ==
    if comparator is None:
        comparator = lambda p, g: (p == g)

    N_total = len(extracted_list_for_all_datapoints)
    assert all(len(row) == 6 for row in extracted_list_for_all_datapoints), "extracted는 (N,6) 구조여야 함"

    # 부분 선택(네 시각화 함수와 동일한 방식)
    if indices is not None:
        sel = list(indices)
    else:
        end = N_total if n is None else min(N_total, start + n)
        sel = list(range(start, end))
    if not sel:
        raise ValueError("선택된 행이 없습니다. start/n 또는 indices를 확인하세요.")

    # (pred, score)에서 pred만 꺼내기 + 보기 좋은 문자열로 포맷
    def get_pred(x):
        if isinstance(x, (tuple, list)) and len(x) >= 1:
            return x[0]
        return x

    def fmt(v):
        if v is None:
            return none_token
        # 정수처럼 보이는 float는 깔끔하게
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)

    rows = []
    for i in sel:
        g = gt_list_for_all_datapoints[i]
        # 6개 블록의 예측값
        preds = [get_pred(extracted_list_for_all_datapoints[i][j]) for j in range(6)]
        extracted_answers_str = ", ".join(fmt(p) for p in preds)

        # 정답 맞춘 블록 라벨
        labels = []
        if g is not None:
            for j, label in enumerate(block_labels):
                p = preds[j]
                if p is not None and comparator(p, g):
                    labels.append(label)
        correct_blocks_str = ", ".join(labels) if labels else "—"

        rows.append({
            "gt": fmt(g),
            f"{block_labels}_extracted_answers": extracted_answers_str,
            "correct_blocks": correct_blocks_str,
            # 필요하면 개수도 추가 가능: "num_correct": len(labels)
        })

    df = pd.DataFrame(rows, columns=["gt",f"{block_labels}_extracted_answers","correct_blocks"])
    return df


import pandas as pd

def build_correct_blocks_table_tokens(
    extracted_list_for_all_datapoints,   # (N, 6) with items like (pred, score)
    gt_list_for_all_datapoints,          # (N,) -> pred or None
    token_nl_list_for_all_datapoints,    # (N,) -> (list of NL answers, list of NL tokens)
    block_labels=("8","16","32","64","128","256"),
    start=0, n=None, indices=None,
    comparator=None,                     # 예: math면 lambda p,g: is_equiv(p,g)
    none_token="None"                    # None을 어떻게 표기할지
):
    # 기본 비교: gsm처럼 ==
    if comparator is None:
        comparator = lambda p, g: (p == g)

    N_total = len(extracted_list_for_all_datapoints)
    assert all(len(row) == 6 for row in extracted_list_for_all_datapoints), "extracted는 (N,6) 구조여야 함"

    # 부분 선택(네 시각화 함수와 동일한 방식)
    if indices is not None:
        sel = list(indices)
    else:
        end = N_total if n is None else min(N_total, start + n)
        sel = list(range(start, end))
    if not sel:
        raise ValueError("선택된 행이 없습니다. start/n 또는 indices를 확인하세요.")

    # (pred, score)에서 pred만 꺼내기 + 보기 좋은 문자열로 포맷
    def get_pred(x):
        if isinstance(x, (tuple, list)) and len(x) >= 1:
            return x[0]
        return x

    def fmt(v):
        if v is None:
            return none_token
        # 정수처럼 보이는 float는 깔끔하게
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)

    rows = []
    for i in sel:
        g = gt_list_for_all_datapoints[i]
        # 6개 블록의 예측값
        preds = [get_pred(extracted_list_for_all_datapoints[i][j]) for j in range(6)]
        extracted_answers_str = ", ".join(fmt(p) for p in preds)

        # 정답 맞춘 블록 라벨
        labels = []
        if g is not None:
            for j, label in enumerate(block_labels):
                p = preds[j]
                if p is not None and comparator(p, g):
                    labels.append(label)
        correct_blocks_str = ", ".join(labels) if labels else "—"

        # 자연어 정보
        nl_answers, nl_tokens, question = token_nl_list_for_all_datapoints[i]

        rows.append({
            'question': question,
            "gt": fmt(g),
            f"{block_labels}_extracted_answers": extracted_answers_str,
            "correct_blocks": correct_blocks_str,
            "nl_answers": nl_answers,
            "nl_tokens": nl_tokens,
        })

    df = pd.DataFrame(rows, columns=['question',
                                    "gt",
                                     f"{block_labels}_extracted_answers",
                                     "correct_blocks",
                                     "nl_answers",
                                     "nl_tokens"])
    return df

def make_answer_grid_from_tuples_3(
    extracted_list_for_all_datapoints,   # (N, 6) with (pred, score)
    majvoted_list_for_all_datapoints,    # (N, 1) -> [pred] 또는 (N,) -> pred
    gt_list_for_all_datapoints,          # (N,) -> pred or None
    xlabels=("8","16","32","64","128","256","majvote","gt"),
    savepath=None,
    show=True,
    start=0,
    n=None,
    indices=None,
    # 점수 계산을 너 코드에 맞추기 위한 비교 함수 훅 (기본은 ==)
    comparator=None    # 예: lambda p,g: is_equiv(p,g)  (math일 때), lambda p,g: evaluate_countdown(p,g) (countdown일 때)
):
    # 기본 비교 (gsm과 동일)
    if comparator is None:
        comparator = lambda p, g: (p == g)

    N_total = len(extracted_list_for_all_datapoints)
    assert len(majvoted_list_for_all_datapoints) == N_total
    assert len(gt_list_for_all_datapoints) == N_total
    assert all(len(row) == 6 for row in extracted_list_for_all_datapoints)

    # --- 서브셋 선택 ---
    if indices is not None:
        sel = list(indices)
    else:
        end = N_total if n is None else min(N_total, start + n)
        sel = list(range(start, end))

    print("DEBUG sel:", len(sel), sel[:20])
    if not sel:
        raise ValueError("선택된 행이 없습니다. start/n 또는 indices를 확인하세요.")

    # 헬퍼
    def get_pred(x):
        if isinstance(x, (tuple, list)) and len(x) >= 1:
            return x[0]
        return x

    # 선택된 부분만 추출
    ext = [extracted_list_for_all_datapoints[i] for i in sel]
    # majvote가 [pred] 또는 pred일 수 있어서 평탄화
    maj = []
    for i in sel:
        row = majvoted_list_for_all_datapoints[i]
        if isinstance(row, (list, tuple)):
            maj.append(row[0] if len(row) > 0 else None)
        else:
            maj.append(row)
    gt  = [gt_list_for_all_datapoints[i] for i in sel]

    N = len(sel)
    grid_rgb = np.ones((N, 8, 3), dtype=float)

    WHITE = [1, 1, 1]
    BLACK = [0, 0, 0]
    BLUE  = [0, 0, 1]

    # ---- 색칠: None → 검정, GT열 기본 파랑 ----
    for i in range(N):
        # 앞 6개 방법
        for j in range(6):
            pred = get_pred(ext[i][j])
            if pred is None:
                grid_rgb[i, j] = WHITE      # ← None은 검정
        # majvote
        if maj[i] is None:
            grid_rgb[i, 6] = WHITE
        # gt
        if gt[i] is None:
            grid_rgb[i, 7] = BLACK
        else:
            grid_rgb[i, 7] = BLUE

    # ---- GT와 비교: 같으면 파랑, 다르면 하양 ----
    for i in range(N):
        g = gt[i]
        for j in range(6):
            pred = get_pred(ext[i][j])
            if pred is None:
                continue
            grid_rgb[i, j] = BLUE if (g is not None and comparator(pred, g)) else WHITE
        if maj[i] is not None:
            grid_rgb[i, 6] = BLUE if (g is not None and comparator(maj[i], g)) else WHITE

    # ---- 열별 정확도 계산 ----
    # 너 코드(printed accuracy)와 동일하게: 분모 = 선택된 전체 N (gt가 None인 행도 포함, 이 행은 무조건 오답)
    denom = max(N, 1)

    acc = []
    # 6개 방법
    for j in range(6):
        correct = 0
        for i in range(N):
            g = gt[i]
            pred = get_pred(ext[i][j])
            if (g is not None) and (pred is not None) and comparator(pred, g):
                correct += 1
            # g is None 이거나 pred is None 은 오답으로 카운트 (분모엔 포함)
        acc.append(100.0 * correct / denom)

    # majvote
    correct = 0
    for i in range(N):
        g = gt[i]
        p = maj[i]
        if (g is not None) and (p is not None) and comparator(p, g):
            correct += 1
    acc.append(100.0 * correct / denom)

    # gt열(자기 자신 비교)은 의미 없으니 표시는 — 로
    acc.append(np.nan)

    # ---- 플롯: 그리드 + 아래 퍼센트 ----
    dpi = 300
    fig = plt.figure(figsize=(8, max(4, min(16, int(N/40)))), dpi=dpi)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[10, 1], hspace=0.01)

    ax = fig.add_subplot(gs[0])
    ax.imshow(grid_rgb, aspect='auto', interpolation='nearest')
    ax.set_xticks(range(8))
    ax.set_xticklabels(xlabels)
    ax.set_yticks([])
    ax.set_xlabel("Methods")
    ax.set_title("Answers (rows={} / total={}) — blue=match GT, white=mismatch, black=GT is null"
                 .format(N, N_total))

    # 얇은 그리드
    ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    # ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.3)
    ax.tick_params(which='minor', bottom=False, left=False)

    # 아래 퍼센트 영역
    axb = fig.add_subplot(gs[1], sharex=ax)
    axb.set_xlim(-0.5, 7.5)
    axb.set_ylim(0, 1)
    axb.axis('off')

    for j in range(8):
        label = "—" if j == 7 or (isinstance(acc[j], float) and np.isnan(acc[j])) else f"{acc[j]:.2f}%"
        axb.text(j, 0.5, label, ha='center', va='center', fontsize=10)

    # 범례
    legend_patches = [
        Patch(facecolor='black', edgecolor='black', label='None'),
        Patch(facecolor='white', edgecolor='black', label='Mismatch'),
        Patch(facecolor='blue',  edgecolor='black', label='Match'),
    ]
    ax.legend(handles=legend_patches, loc='upper right', frameon=True)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def make_answer_grid_general(
    extracted_list_for_all_datapoints,     # (N, M) with (pred, score) or just pred
    majvoted_list_for_all_datapoints,      # (N,) or (N,1)
    gt_list_for_all_datapoints,            # (N,)
    xlabels=None,                           # e.g., ("8","16","32","64","128","256","majvote","gt")
    savepath=None,
    show=True,
    start=0,
    n=None,
    indices=None,
    comparator=None      # e.g., lambda p,g: p==g
):
    """세로형(행=데이터포인트, 열=방법들+majvote+gt). 열 개수 가변 지원."""
    # 기본 비교
    if comparator is None:
        comparator = lambda p, g: (p == g)

    N_total = len(extracted_list_for_all_datapoints)
    assert len(majvoted_list_for_all_datapoints) == N_total
    assert len(gt_list_for_all_datapoints) == N_total
    # 방법 개수(M)를 첫 행에서 추론
    M = len(extracted_list_for_all_datapoints[0])
    assert all(len(row) == M for row in extracted_list_for_all_datapoints), "모든 행의 방법 개수가 동일해야 합니다."

    # 서브셋 선택
    if indices is not None:
        sel = list(indices)
    else:
        end = N_total if n is None else min(N_total, start + n)
        sel = list(range(start, end))
    if not sel:
        raise ValueError("선택된 행이 없습니다. start/n 또는 indices를 확인하세요.")

    # 헬퍼: (pred,score) → pred
    def get_pred(x):
        if isinstance(x, (tuple, list)) and len(x) >= 1:
            return x[0]
        return x

    # 선택된 데이터만 추출
    ext = [extracted_list_for_all_datapoints[i] for i in sel]
    maj = []
    for i in sel:
        row = majvoted_list_for_all_datapoints[i]
        if isinstance(row, (list, tuple)):
            maj.append(row[0] if len(row) > 0 else None)
        else:
            maj.append(row)
    gt  = [gt_list_for_all_datapoints[i] for i in sel]

    N = len(sel)

    # 색상 매핑
    WHITE = np.array([1., 1., 1.])
    BLACK = np.array([0., 0., 0.])
    BLUE  = np.array([0., 0., 1.])

    # 그리드: N x (M + 2)  (methods..., majvote, gt)
    grid_rgb = np.ones((N, M + 2, 3), dtype=float)

    # 기본 배경 (예전 로직 유지: pred None/maj None → 흰색, gt None → 검정, gt 채워진 칸은 파랑)
    for i in range(N):
        # methods
        for j in range(M):
            pred = get_pred(ext[i][j])
            if pred is None:
                grid_rgb[i, j] = WHITE
        # majvote
        if maj[i] is None:
            grid_rgb[i, M] = WHITE
        # gt
        if gt[i] is None:
            grid_rgb[i, M + 1] = BLACK
        else:
            grid_rgb[i, M + 1] = BLUE

    # GT 비교 결과로 색칠 (일치: 파랑, 불일치: 흰색)
    for i in range(N):
        g = gt[i]
        for j in range(M):
            pred = get_pred(ext[i][j])
            if pred is None:
                continue
            grid_rgb[i, j] = BLUE if (g is not None and comparator(pred, g)) else WHITE
        if maj[i] is not None:
            grid_rgb[i, M] = BLUE if (g is not None and comparator(maj[i], g)) else WHITE

    # 정확도
    denom = max(N, 1)
    acc = []
    for j in range(M):
        correct = 0
        for i in range(N):
            g = gt[i]
            pred = get_pred(ext[i][j])
            if (g is not None) and (pred is not None) and comparator(pred, g):
                correct += 1
        acc.append(100.0 * correct / denom)

    # majvote
    correct = 0
    for i in range(N):
        g = gt[i]
        p = maj[i]
        if (g is not None) and (p is not None) and comparator(p, g):
            correct += 1
    acc.append(100.0 * correct / denom)

    # gt 열은 NaN
    acc.append(np.nan)

    # 라벨
    if xlabels is None:
        base = [str(x) for x in range(M)]
        xlabels = tuple(base + ["majvote", "gt"])
    else:
        assert len(xlabels) == M + 2, f"xlabels 길이는 {M+2} 이어야 합니다."

    # 플롯(세로형)
    dpi = 300
    fig = plt.figure(figsize=(max(6, min(18, (M+2))), max(4, min(16, int(N / 40)))), dpi=dpi)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[10, 1], hspace=0.08)

    ax = fig.add_subplot(gs[0])
    ax.imshow(grid_rgb, aspect='auto', interpolation='nearest')
    ax.set_xticks(range(M + 2))
    ax.set_xticklabels(xlabels)
    ax.set_yticks([])
    ax.set_xlabel("Methods")
    ax.set_title(f"Answers (rows={N} / total={N_total}) — blue=match GT, white=mismatch, black=GT is null")

    # 아래 퍼센트
    axb = fig.add_subplot(gs[1], sharex=ax)
    axb.set_xlim(-0.5, M + 1.5)
    axb.set_ylim(0, 1)
    axb.axis('off')
    for j in range(M + 2):
        label = "—" if j == (M + 1) or (isinstance(acc[j], float) and np.isnan(acc[j])) else f"{acc[j]:.2f}%"
        axb.text(j, 0.5, label, ha='center', va='center', fontsize=10)

    # 범례
    legend_patches = [
        Patch(facecolor='black', edgecolor='black', label='GT None'),
        Patch(facecolor='white', edgecolor='black', label='Mismatch/None'),
        Patch(facecolor='blue',  edgecolor='black', label='Match'),
    ]
    ax.legend(handles=legend_patches, loc='upper right', frameon=True)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def make_answer_grid_horizontal(
    extracted_list_for_all_datapoints,
    majvoted_list_for_all_datapoints,
    gt_list_for_all_datapoints,
    ylabels=None,                         # 방법 라벨(세로축). None이면 0..M-1 + ["majvote","gt"]
    savepath=None,
    show=True,
    start=0,
    n=None,
    indices=None,
    comparator=None
):
    """가로형(열=데이터포인트, 행=방법들+majvote+gt). 긴 세로 대신 가로로 넓게."""
    if comparator is None:
        comparator = lambda p, g: (p == g)

    N_total = len(extracted_list_for_all_datapoints)
    assert len(majvoted_list_for_all_datapoints) == N_total
    assert len(gt_list_for_all_datapoints) == N_total
    M = len(extracted_list_for_all_datapoints[0])
    assert all(len(row) == M for row in extracted_list_for_all_datapoints)

    # 선택
    if indices is not None:
        sel = list(indices)
    else:
        end = N_total if n is None else min(N_total, start + n)
        sel = list(range(start, end))
    if not sel:
        raise ValueError("선택된 행이 없습니다. start/n 또는 indices를 확인하세요.")

    def get_pred(x):
        if isinstance(x, (tuple, list)) and len(x) >= 1:
            return x[0]
        return x

    ext = [extracted_list_for_all_datapoints[i] for i in sel]
    maj = []
    for i in sel:
        row = majvoted_list_for_all_datapoints[i]
        if isinstance(row, (list, tuple)):
            maj.append(row[0] if len(row) > 0 else None)
        else:
            maj.append(row)
    gt  = [gt_list_for_all_datapoints[i] for i in sel]

    N = len(sel)

    WHITE = np.array([1., 1., 1.])
    BLACK = np.array([0., 0., 0.])
    BLUE  = np.array([0., 0., 1.])

    # (M + 2) x N
    grid_rgb = np.ones((M + 2, N, 3), dtype=float)

    # 기본 색
    for j in range(M):              # 행: 방법 index
        for i in range(N):          # 열: 데이터 index
            pred = get_pred(ext[i][j])
            if pred is None:
                grid_rgb[j, i] = WHITE
    for i in range(N):
        if maj[i] is None:
            grid_rgb[M, i] = WHITE
        if gt[i] is None:
            grid_rgb[M + 1, i] = BLACK
        else:
            grid_rgb[M + 1, i] = BLUE

    # GT 비교
    for j in range(M):
        for i in range(N):
            pred = get_pred(ext[i][j])
            g = gt[i]
            if pred is None:
                continue
            grid_rgb[j, i] = BLUE if (g is not None and comparator(pred, g)) else WHITE
    for i in range(N):
        g = gt[i]
        p = maj[i]
        if p is not None:
            grid_rgb[M, i] = BLUE if (g is not None and comparator(p, g)) else WHITE

    # 정확도 (분모=N)
    denom = max(N, 1)
    acc = []
    for j in range(M):
        correct = 0
        for i in range(N):
            g = gt[i]
            pred = get_pred(ext[i][j])
            if (g is not None) and (pred is not None) and comparator(pred, g):
                correct += 1
        acc.append(100.0 * correct / denom)
    # majvote
    correct = 0
    for i in range(N):
        g = gt[i]
        p = maj[i]
        if (g is not None) and (p is not None) and comparator(p, g):
            correct += 1
    acc.append(100.0 * correct / denom)
    # gt는 NaN
    acc.append(np.nan)

    # 라벨
    if ylabels is None:
        ylabels = [str(k) for k in range(M)] + ["majvote", "gt"]
    else:
        assert len(ylabels) == M + 2, f"ylabels 길이는 {M+2} 이어야 합니다."

    # 플롯(가로형: 행=방법, 열=데이터)
    dpi = 300
    # 가로로 길게: 폭은 N에 비례, 높이는 방법 수에 비례
    fig_w = max(8, min(24, int(N / 40) + 8))
    fig_h = max(4, min(16, int((M + 2) / 3) + 2))
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[10, 1], wspace=0.05)

    ax = fig.add_subplot(gs[0])
    ax.imshow(grid_rgb, aspect='auto', interpolation='nearest')
    ax.set_yticks(range(M + 2))
    ax.set_yticklabels(ylabels)
    ax.set_xticks([])     # 데이터포인트가 너무 많으면 축 제거
    ax.set_xlabel(f"Datapoints (N={N} / total={N_total})")
    ax.set_title("Answers — blue=match GT, white=mismatch, black=GT is null")

    # 오른쪽에 정확도 표기
    axr = fig.add_subplot(gs[1], sharey=ax)
    axr.set_ylim(-0.5, M + 1.5)
    axr.set_xlim(0, 1)
    axr.axis('off')
    for j in range(M + 2):
        label = "—" if j == (M + 1) or (isinstance(acc[j], float) and np.isnan(acc[j])) else f"{acc[j]:.2f}%"
        axr.text(0.5, j, label, ha='center', va='center', fontsize=10)

    legend_patches = [
        Patch(facecolor='black', edgecolor='black', label='GT None'),
        Patch(facecolor='white', edgecolor='black', label='Mismatch/None'),
        Patch(facecolor='blue',  edgecolor='black', label='Match'),
    ]
    ax.legend(handles=legend_patches, loc='upper right', frameon=True)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def make_answer_grid_horizontal(
    extracted_list_for_all_datapoints,
    majvoted_list_for_all_datapoints,   # 사용하지 않지만 시그니처 유지
    gt_list_for_all_datapoints,
    ylabels=None,
    savepath=None,
    show=True,
    start=0,
    n=None,
    indices=None,
    comparator=None
):
    if comparator is None:
        comparator = lambda p, g: (p == g)

    N_total = len(extracted_list_for_all_datapoints)
    M = len(extracted_list_for_all_datapoints[0])
    assert all(len(row) == M for row in extracted_list_for_all_datapoints)

    if indices is not None:
        sel = list(indices)
    else:
        end = N_total if n is None else min(N_total, start + n)
        sel = list(range(start, end))
    if not sel:
        raise ValueError("선택된 행이 없습니다.")

    def get_pred(x):
        if isinstance(x, (tuple, list)) and len(x) >= 1:
            return x[0]
        return x

    ext = [extracted_list_for_all_datapoints[i] for i in sel]
    gt  = [gt_list_for_all_datapoints[i] for i in sel]
    N = len(sel)

    WHITE = np.array([1., 1., 1.])
    BLACK = np.array([0., 0., 0.])
    PASTEL_BLUE = np.array([0.5, 0.7, 1.0])   # 매치(정답)용 파스텔 블루
    PURPLE = np.array([0.7, 0.5, 0.9])        # "모두 오답(비GT)" 강조용 보라색

    grid_rgb = np.ones((M + 1, N, 3), dtype=float)

    # 초기 색상
    for j in range(M):
        for i in range(N):
            pred = get_pred(ext[i][j])
            if pred is None:
                grid_rgb[j, i] = WHITE
    for i in range(N):
        if gt[i] is None:
            grid_rgb[M, i] = BLACK
        else:
            grid_rgb[M, i] = PASTEL_BLUE

    # GT 매칭 색상
    for j in range(M):
        for i in range(N):
            pred = get_pred(ext[i][j])
            g = gt[i]
            if pred is None:
                continue
            grid_rgb[j, i] = PASTEL_BLUE if (g is not None and comparator(pred, g)) else WHITE

    # "모두 오답(비GT)" 열 찾기:
    # 조건: j=0..M-1 모든 셀 색이 WHITE (즉, 어떤 블루 매치도 없음) 이고 GT는 None이 아님
    for i in range(N):
        if gt[i] is not None:
            # 모든 비GT 행이 흰색인지 체크
            col_block = grid_rgb[0:M, i]
            all_white = np.allclose(col_block, WHITE)
            if all_white:
                # 해당 열의 GT 셀을 보라색으로 덮어 강조
                grid_rgb[M, i] = PURPLE

    # 정확도 계산
    denom = max(N, 1)
    acc = []
    for j in range(M):
        correct = 0
        for i in range(N):
            g = gt[i]
            pred = get_pred(ext[i][j])
            if (g is not None) and (pred is not None) and comparator(pred, g):
                correct += 1
        acc.append(100.0 * correct / denom)
    acc.append(np.nan)

    if ylabels is None:
        ylabels = [str(k) for k in range(M)] + ["GT"]
    else:
        ylabels = [lab for lab in ylabels if lab.lower() != "majvote"]
        assert len(ylabels) == M + 1

    dpi = 300
    fig_w = max(8, min(24, int(N / 40) + 8))
    fig_h = max(4, min(16, int((M + 1) / 3) + 2))
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[10, 1], wspace=0.05)

    ax = fig.add_subplot(gs[0])
    ax.imshow(grid_rgb, aspect='auto', interpolation='nearest')
    ax.set_yticks(range(M + 1))
    ax.set_yticklabels(ylabels, fontsize=25)
    ax.set_xticks([0, N_total - 1])
    ax.set_xticklabels([0, N_total - 1], fontsize=25)
    ax.set_xlabel(f"GSM8K datapoints ({N_total})", fontsize=30, labelpad=30)
    ax.set_ylabel("Block Size", fontsize=30, labelpad=15)
    ax_right = ax.twinx()
    ax_right.set_ylabel("Accuracy (%)", fontsize=30, rotation=270, labelpad=200)
    ax_right.set_yticks([])
    ax.set_title("Answer correctness per datapoint", fontsize=35, pad=15)

    # 오른쪽 accuracy 표기
    axr = fig.add_subplot(gs[1], sharey=ax)
    axr.set_ylim(-0.5, M + 0.5)
    axr.set_xlim(0, 1)
    axr.axis('off')

    for j in range(M + 1):
        label = "100.00%" if j == M or (isinstance(acc[j], float) and np.isnan(acc[j])) else f"{acc[j]:.2f}%"
        axr.text(0.5, j, label, ha='center', va='center', fontsize=25)

    # 범례: Match / Mismatch / GT is None / 모두 오답(비GT)
    legend_patches = [
        Patch(facecolor=PASTEL_BLUE, edgecolor='black', label='Match'),
        Patch(facecolor='white', edgecolor='black', label='Mismatch'),
        Patch(facecolor='black', edgecolor='black', label='GT is None'),
        Patch(facecolor=PURPLE, edgecolor='black', label='All wrong (non-GT)'),
    ]
    ax.legend(
        handles=legend_patches,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.25),
        frameon=True,
        fontsize=20,
        ncol=len(legend_patches),
        handlelength=1.5,
        columnspacing=1.5
    )

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)



import re
if __name__ == "__main__":
  
    ################
    folder_path = '/home/work/jihoon_wombat_storage/JIHOON/d1_llada_grpo_test/eval/eval_result_20250831_arcc_aime_truthful'
    folder_path = '/home/work/jihoon_wombat_storage/JIHOON/d1_llada_grpo_test/eval/eval_result_20250922_128_[4, 8, 16, 32, 64]_42'
    # folder_path = '/home/work/jihoon_wombat_storage/JIHOON/d1_llada_grpo_test/eval/eval_result_20250916_dynamic_increasing_blocksize_BOLT'
    # folder_path = '/home/work/jihoon_wombat_storage/JIHOON/d1_llada_grpo_test/eval/eval_result_20250909_truthfulqa_subset_BOLT/seed42_bolt'
    # folder_path = '/home/work/jihoon_wombat_storage/JIHOON/d1_llada_grpo_test/eval/eval_result_20250831_arcc_aime_truthful'
    
    
    folder_path = '/home/work/jihoon_wombat_storage/JIHOON/d1_llada_grpo_test/eval/eval_result_20250923_512_random_topkmargin'
    data_type = 'truthfulqa'
    ###############

    select_strategy = 'select_first' # 'either_right' 'select_first'
    vote_rule = 'frequency' # 'frequency_tie_lowest_nll' 'lowest_nll' 'frequency'

    save_BOLT_json = False

    # include_str_list = ['block8', 'block16', 'block32', 'block64', 'block128', 'block256']
    # include_str_list = ['block16', 'block32', 'block64', 'block128']
    # include_str_list = [f'[{8*i}, {256-(8*i)}]' for i in range(1,26)]

    include_str_list = ['_seed0_','_seed1_','_seed2_','_seed3_','_seed4_','_seed42_'] # [] means just use all json in that folder: excluding nothing
    include_str_list = ['block8', 'block16', 'block32', 'block64', 'block128']


    include_str_list = ['_seed42_', '_seed0_','_seed1_','_seed2_','_seed3_','_seed4_']
    # include_str_list = ['_seed42_']
    # BOLT for MMLU
    include_str_list = ['/home/work/jihoon_wombat_storage/JIHOON/d1_llada_grpo_test/eval/eval_result_20250908_mmlu_BOLT/20250909_192758_mmlu_LLaDA-8B-Instruct_low_confidence_genlen256_diffsteps128_block8_seed42_kvbaseline_generations.json',
                        '/home/work/jihoon_wombat_storage/JIHOON/d1_llada_grpo_test/eval/eval_result_20250908_mmlu_BOLT/20250909_194143_mmlu_LLaDA-8B-Instruct_low_confidence_genlen256_diffsteps128_block16_seed42_kvbaseline_generations.json',
                        '/home/work/jihoon_wombat_storage/JIHOON/d1_llada_grpo_test/eval/eval_result_20250908_mmlu_BOLT/20250909_195528_mmlu_LLaDA-8B-Instruct_low_confidence_genlen256_diffsteps128_block32_seed42_kvbaseline_generations.json',
                        '/home/work/jihoon_wombat_storage/JIHOON/d1_llada_grpo_test/eval/eval_result_20250908_mmlu_BOLT/20250909_200913_mmlu_LLaDA-8B-Instruct_low_confidence_genlen256_diffsteps128_block64_seed42_kvbaseline_generations.json',
                        '/home/work/jihoon_wombat_storage/JIHOON/d1_llada_grpo_test/eval/eval_result_20250908_mmlu_BOLT/20250909_202256_mmlu_LLaDA-8B-Instruct_low_confidence_genlen256_diffsteps128_block128_seed42_kvbaseline_generations.json',]
    



    include_str_list = []
    exclude_str_list = ['/home/work/jihoon_wombat_storage/JIHOON/d1_llada_grpo_test/eval/eval_result_20250923_512_random_topkmargin/20250924_033242_truthfulqa_LLaDA-8B-Instruct_topk_margin_genlen512_diffsteps256_block512_seed42_kvbaseline_generations.json']
    

    a = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            abs_path = os.path.abspath(os.path.join(root, file))
            if data_type in abs_path:
                a.append(abs_path)
    include_str_list = a[:] # a[-30:], a[-25:], a[-20:], a[-15:], a[-10:],a[-5:] for 


    # print(include_str_list)
    ###############################
    # Dynamic BOLT check
    ###############################
    # a = []
    # for root, dirs, files in os.walk(folder_path):
    #     for file in files:
    #         abs_path = os.path.abspath(os.path.join(root, file))
    #         a.append(abs_path)
    # include_str_list = a[-30:] # a[-30:], a[-25:], a[-20:], a[-15:], a[-10:],a[-5:] for 
    # # '/home/work/jihoon_wombat_storage/JIHOON/d1_llada_grpo_test/eval/eval_result_20250916_dynamic_increasing_blocksize_BOLT'
    # exclude_str_list = []
    ###############################


    
    if save_BOLT_json:
        BOLT_json = {
            "generations": []
        }
        def add_generation(BOLT_json, question, prompt_input, generations, ground_truth, loglikelihood):
            BOLT_json["generations"].append({
                "question": question,
                "prompt_input": prompt_input,
                "generations": generations,
                "ground_truth": ground_truth,
                "loglikelihood": loglikelihood
            })


    paths = get_all_file_paths_2(folder_path, data_type, include_str_list = include_str_list, exclude_str_list = exclude_str_list)
    for path in paths:
        print(path)
    print(f'majority vote paths count: {len(paths)}')
    print(f'dataset: {data_type} | select_strategy: {select_strategy}')
    all_paths = [[paths[i]] for i in range(len(paths))]+[paths]
    all_paths = [paths] # Just for quick result

    
    for now_target_path_index in tqdm(range(len(all_paths))):
        paths = all_paths[now_target_path_index]
        # 모든 파일 미리 읽어오기
        all_data = []
        for path in paths:
            with open(path, encoding="utf-8") as f:
                all_data.append(json.load(f))

        # 기준 파일 (첫 번째 파일 ground_truth 사용)
        base_data = all_data[0]
        num_datapoints = len(base_data["generations"])

        result_list = []
        result_tie_list = []
        # distance_list = []
        # common_prefix_length_list = [] 
        # generation_length_list = []



        extracted_list_for_all_datapoints = []
        extracted_list_for_all_datapoints_for_BOLT = []
        majvoted_list_for_all_datapoints = [] 
        gt_list_for_all_datapoints = []
        token_nl_list_for_all_datapoints = []
        nll_list = []

        for num in range(num_datapoints):
            now_datapoint_question = base_data["generations"][num]["prompt_input"]
            now_datapoint_gt = base_data["generations"][num]["ground_truth"]
            now_datapoint_question_0 = base_data["generations"][num]["question"]
            now_extracted_ans_list = []
            now_ans_list = []
            now_nl_tokens_list = []

            now_generation_and_extraction = []


            for data in all_data:
            
                now_generation = data["generations"][num]["generations"]
                now_question = data["generations"][num]["question"]



                try:
                    now_generation_ll = data["generations"][num]["loglikelihood"]
                except Exception:
                    now_generation_ll = 0     
                
                try:
                    now_generation_nl_tokens = data['generations'][num]["intermediate_types"][-1]
                    now_nl_tokens_list.append(now_generation_nl_tokens)
                   
                except Exception:
                    pass        

                

                if data_type == 'gsm':
                    now_extracted = gsm_extract_ans_from_generation(now_generation)
                elif data_type == 'math':
                    now_extracted = parse_math_answers(now_generation)
                elif data_type == 'countdown':
                    now_extracted = parse_countdown_answers(now_generation)
                elif data_type == 'sudoku':
                    now_extracted = parse_sudoku_answers(now_question, now_generation)
                elif data_type == 'arcc':
                    now_extracted = parse_math_answers(now_generation)
                elif data_type == 'truthfulqa':
                    now_extracted = parse_math_answers(now_generation)
                elif data_type == 'hellaswag':
                    now_extracted = parse_math_answers(now_generation)
                elif data_type == 'mmlu':
                    now_extracted = parse_math_answers(now_generation)



                now_ans_list.append(now_generation)
                now_extracted_ans_list.append((now_extracted, now_generation_ll))
                now_generation_and_extraction.append((now_extracted, now_generation, now_generation_ll))


            extracted_list_for_all_datapoints.append(now_extracted_ans_list)
            token_nl_list_for_all_datapoints.append((now_ans_list, now_nl_tokens_list, now_datapoint_question))
            extracted_list_for_all_datapoints_for_BOLT.append(now_generation_and_extraction)

            # print(now_nl_tokens_list)
        

            # print(now_extracted_ans_list, now_datapoint_gt)
            # majority_voted, is_tied, distance = most_frequent_element_4(now_extracted_ans_list)
            # if distance: distance_list.append(distance)

            # if num == 3:
            #     print(now_ans_list)
            #     exit()

            # ######################  !!!!!!!!!!!!!!!!!!   ################################
            # majority_voted_list, is_tied = most_frequent_elements(now_extracted_ans_list) 
            # # Past selecting criteria. None can be the final output!!
            # # But our prime rule of majvote is that 'we vote for extracted answer' so lets exclude None 
            # majority_voted_list, is_tied = most_frequent_elements_exclude_None(now_extracted_ans_list) 


            # print(now_extracted_ans_list)
    
            majority_voted_list, is_tied = most_frequent_elements_exclude_None_and_tie_lownll(now_extracted_ans_list, 
            selecting_rule=vote_rule) # 'frequency_tie_lowest_nll' 'lowest_nll' 'frequency'


            # if is_tied:
            #     print(majority_voted_list)
            # print(majority_voted_list)


            if select_strategy == 'select_first': # This is out BOLT
                majority_voted_list = majority_voted_list[:1] # for select first one when tied
                # print(majority_voted_list)
                nll = majority_voted_list[0][1] # (voted one ans, nll for that)
                nll_list.append(nll)
                

                ##################################################
                # Generate BOLT json 
                if save_BOLT_json:
                    bolt_selected_generation = None
                    for entry in extracted_list_for_all_datapoints_for_BOLT:
                        for extracted, generation, ll in entry:
                            if majority_voted_list[0][0] == extracted and nll == -ll:
                                bolt_selected_generation = generation
                                break
                    
                    assert bolt_selected_generation != None 
                    add_generation(BOLT_json, now_datapoint_question_0, now_datapoint_question, bolt_selected_generation, now_datapoint_gt, -nll)
                ##################################################


            elif select_strategy == 'either_right':
                majority_voted_list = majority_voted_list 
                nll = majority_voted_list[0][1] # (voted one ans, nll for that)
                nll_list.append(nll)
            # print(majority_voted_list)

            majvoted_list_for_all_datapoints.append(majority_voted_list)
            gt_list_for_all_datapoints.append(now_datapoint_gt)
            # print("모든 문자열이 동일한 prefix 길이:", common_prefix_length)

            is_corrct_list = []
            sudoku_list = []
    

        

            for majority_voted, voted_ans_nll in majority_voted_list:

                
                if data_type == 'gsm':
                    is_correct = False
                    if majority_voted is not None:
                        is_correct = majority_voted == now_datapoint_gt
                    else: 
                        is_correct = False
                    is_corrct_list.append(is_correct)

                    # if is_correct_gsm:
                    #     result_list.append(1)
                    # else:
                    #     result_list.append(0)

                elif data_type == 'math':
                    is_correct = False
                    if majority_voted is not None:
                        is_correct = is_equiv(majority_voted, now_datapoint_gt)
                    is_corrct_list.append(is_correct)

                    # if is_correct_math:
                    #     result_list.append(1)
                    # else:
                    #     result_list.append(0)

                elif data_type == 'countdown':
                    is_correct = evaluate_countdown(majority_voted, now_datapoint_gt)
                    is_corrct_list.append(is_correct)

                    # if is_correct:
                    #     result_list.append(1)
                    # else:
                    #     result_list.append(0)

                elif data_type == 'sudoku':
                    numerator_count, denominator_count = evaluate_sudoku_answers(majority_voted, now_question, now_datapoint_gt)
                    sudoku_list.append((numerator_count, denominator_count))

                elif data_type == 'arcc':
                    is_correct = False
                    if majority_voted is not None:
                        is_correct = is_equiv(majority_voted, now_datapoint_gt)
                    is_corrct_list.append(is_correct)


                elif data_type == 'truthfulqa':
                    is_correct = False
                    if majority_voted is not None:
                        is_correct = is_equiv(majority_voted, now_datapoint_gt)
                    is_corrct_list.append(is_correct)

                elif data_type == 'hellaswag':
                    is_correct = False
                    if majority_voted is not None:
                        is_correct = is_equiv(majority_voted, now_datapoint_gt)
                    is_corrct_list.append(is_correct)
                
                elif data_type == 'mmlu':
                    is_correct = False
                    if majority_voted is not None:
                        is_correct = is_equiv(majority_voted, now_datapoint_gt)
                    is_corrct_list.append(is_correct)

            # print(is_corrct_list)


            if data_type in ['gsm', 'math', 'countdown', 'arcc', 'truthfulqa', 'hellaswag', 'mmlu']:
                # print(f'{num} | {now_extracted_ans_list} | {majority_voted_list} | {now_datapoint_gt}')
                if True in is_corrct_list: result_list.append(1)
                else: result_list.append(0)
            elif data_type in ['sudoku']:
                compare_tuples_list = [(numerator_count/denominator_count, numerator_count, denominator_count) for numerator_count, denominator_count in sudoku_list]
                # print(num, compare_tuples_list)
                max_tuple = max(compare_tuples_list, key=lambda x: x[0]) # choose best among tied
                result_list.append((max_tuple[1], max_tuple[2]))
            
            if is_tied: 
                result_tie_list.append(1)
            else:
                result_tie_list.append(0)

        
        # print(f'decoding: {paths}')
        print('-'*30)
        print(f'select_strategy: {select_strategy} | vote_rule: {vote_rule}')

        print(f'majority vote samples count: {len(paths)}')
        if data_type in ['gsm', 'math', 'countdown', 'arcc', 'truthfulqa', 'hellaswag', 'mmlu']:
            print(f'data_type: {data_type}')
            print(f'{(sum(result_list) / len(result_list))*100:.2f}%')
            print(f'{(sum(result_tie_list) / len(result_tie_list))*100:.2f}%')
            # print(f'{sum(distance_list) / len(distance_list) if len(distance_list) != 0 else None}')
            print('-'*30)
        elif data_type in ['sudoku']:
            print(f'data_type: {data_type}')
            all_acc = sum([numerator for numerator, denominator in result_list]) / sum([denominator for numerator, denominator in result_list])
            print(f'{all_acc*100:.2f}%')
            print(f'{(sum(result_tie_list) / len(result_tie_list))*100:.2f}%')
            print('-'*30)

        print(f'nll_list_len: {len(nll_list)}')
        print(f'nll_list_mean: {sum(nll_list) / len(nll_list):.2f}')


        
        
    if data_type == 'gsm': comparator = None 
    elif data_type == 'math' or data_type == 'arcc' or data_type == 'truthfulqa': comparator = lambda p,g: is_equiv(p,g) 
    elif data_type == 'countdown': comparator = lambda p,g: evaluate_countdown(p,g)

    if save_BOLT_json:
        BOLT_json["metrics"] = {
        "wall_time": "sum all wall_time of BOLT samples",
        "total_processed": base_data["metrics"]["total_processed"]
        }
        BOLT_json["model_path"] = base_data["model_path"]
        BOLT_json["checkpoint_path"] = base_data["checkpoint_path"]
        BOLT_json["gen_length"] = base_data["gen_length"] 
        BOLT_json["diffusion_steps"] = "sum all diffusion_steps of BOLT samples"
        BOLT_json["block_length"] = "Used block_lengths of BOLT samples"
        BOLT_json["total_latency"] = "sum all wall_time of BOLT samples"
        BOLT_json["early_ban_steps"] = "Not used for BOLT"

        candidate_folder_name = folder_path.split('/')[-1]
        candidates_count = len(paths)
        BOLT_folder_path = "Z_eval_result_folder_BOLT"  
        if not os.path.exists(BOLT_folder_path): os.makedirs(BOLT_folder_path)
        from datetime import datetime
        now = datetime.now().strftime("%Y%m%d_%H:%M:%S")
        BOLT_filename = f'{now}_BOLT_{candidates_count}_{data_type}_{candidate_folder_name}_generations.json'
        with open(os.path.join(BOLT_folder_path, BOLT_filename), "w", encoding="utf-8") as f: json.dump(BOLT_json, f, indent=2, ensure_ascii=False)


    print(f'BOLT json save: {save_BOLT_json}')

    
    ###########################################################################################
    # 파란색 이미지 만들기
    # make_answer_grid_from_tuples_3(extracted_list_for_all_datapoints,
    #                          majvoted_list_for_all_datapoints,
    #                          gt_list_for_all_datapoints,
    #                          savepath='answer_grid_222.png',
    #                          comparator=comparator,
    #                          n=1319)
    # make_answer_grid_horizontal(
    #     extracted_list_for_all_datapoints, majvoted_list_for_all_datapoints, gt_list_for_all_datapoints,
    #     ylabels=("8","16","32","64","128","256","majvote","GT"),
    #     savepath="grid_horizontal.pdf", show=True
    # )
    # make_answer_grid_horizontal(
    #     extracted_list_for_all_datapoints, majvoted_list_for_all_datapoints, gt_list_for_all_datapoints,
    #     ylabels=("8","16","32","64","128","256","majvote","GT"),
    #     savepath="grid_horizontal.png", show=True
    # )
    ###########################################################################################

    ###########################################################################################
    # # csv 파일 만들기: intermediate steps가 없는 json이 모인 폴더의 경우 
    # import pandas as pd
    # df = build_correct_blocks_table(
    # extracted_list_for_all_datapoints,
    # gt_list_for_all_datapoints,
    # block_labels=("8","16","32","64","128","256"),
    #     # comparator=lambda p,g: is_equiv(p,g)  # math/countdown이면 전달
    # )
    # print(df.head(5))
    # mask = df['correct_blocks'].notna() & df['correct_blocks'].str.strip().ne('—') & df['correct_blocks'].str.strip().ne('')
    # pct = mask.mean() * 100
    # print(pct)
    # # CSV로 저장
    # df.to_csv("correct_blocks_per_datapoint_gsm8k.csv", index=False, encoding="utf-8")
    # # csv 파일 만들기: intermediate steps가 있는 json이 모인 폴더의 경우 
    # df = build_correct_blocks_table_tokens(
    # extracted_list_for_all_datapoints,   # (N, 6) with items like (pred, score)
    # gt_list_for_all_datapoints,          # (N,) -> pred or None
    # token_nl_list_for_all_datapoints,    # (N,) -> (list of NL answers, list of NL tokens)
    # block_labels=("8","16","32","64","128","256"),
    # start=0, n=None, indices=None,
    # comparator=None,                     # 예: math면 lambda p,g: is_equiv(p,g)
    # none_token="None"                    # None을 어떻게 표기할지
    # )
    # print(df.head(5))
    # df.to_csv(f"correct_blocks_per_datapoint_{data_type}_tokens.csv", index=False, encoding="utf-8")
    ###########################################################################################
