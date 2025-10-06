import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

from generate import (
    generate,
    generate_llada,
    generate_with_dual_cache,
    generate_with_prefix_cache,
    generate_DynamicBlock,
    generate_block_search
)

from gsm8k import GSM8KDataset
from math500 import MATH500Dataset
from countdown import CTDDataset
from sudoku import SudokuDataset

from arcc import ARCCDataset
from truthfulqa import TruthfulQAMCDataset

from HEX_run_by_given_json_samples import (
    gsm_extract_ans_from_generation,
    parse_math_answers,
    parse_countdown_answers,
    parse_sudoku_answers,
    most_frequent_elements_exclude_None_and_tie_lownll
)

DATASET_MAP = {
    "gsm8k": GSM8KDataset,
    "math": MATH500Dataset,
    "countdown": CTDDataset,
    "sudoku": SudokuDataset,
    "arcc": ARCCDataset,
    "truthfulqa": TruthfulQAMCDataset,
}

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

def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_ddp():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def evaluate(
    model,
    tokenizer,
    dataloader,
    dataset='',
    gen_length=256,
    temperature=0.0,
    cfg_scale=0.0,
    steps=128,
    block_lengths = [8,16,32,64,128],
    save_intermediate = False,
):
    steps = gen_length // 2
    assert gen_length % 2 == 0, "gen_length % 2 != 0"

    if gen_length == 256: block_lengths = [8,16,32,64,128]
    elif gen_length == 512: block_lengths = [16,32,64,128,256]
    elif gen_length == 128: block_lengths = [4,8,16,32,64]

    model.eval()
    total_processed = torch.tensor(0, device=model.device)
    wall_times = []
    all_generations = []
    device = model.device
    batch_count = 0 
    for batch in tqdm(dataloader, disable=(dist.get_rank() != 0)):
        batch_count += 1
        input_ids = batch["input_ids"].to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]

        HEX_outputs = []
        HEX_batch_candidates = []   # shape: [K][B]
        HEX_batch_lls = []          # shape: [K][B]  <-- 추가

        for now_block in block_lengths:
            if not save_intermediate:
                out, ll = generate(model, input_ids, tokenizer,
                                steps=steps, gen_length=gen_length, block_length=now_block,
                                temperature=temperature, cfg_scale=cfg_scale,
                                remasking='low_confidence', save_intermediate=False)
            else:
                out, ll, intermediates, confidences = generate(model, input_ids, tokenizer,
                                steps=steps, gen_length=gen_length, block_length=now_block,
                                temperature=temperature, cfg_scale=cfg_scale,
                                remasking='low_confidence', save_intermediate=True)

            texts = tokenizer.batch_decode(out[:, -gen_length:], skip_special_tokens=False)  # len B
            HEX_batch_candidates.append(texts)  # append list length B
            HEX_batch_lls.append(ll)            # append list length B  <-- 추가

        # transpose => list length B, each item length K
        HEX_batch_candidates_T = list(map(list, zip(*HEX_batch_candidates)))
        HEX_batch_lls_T = list(map(list, zip(*HEX_batch_lls)))  # <-- 추가

        # 샘플별 집계
        for cand_list_per_sample, ll_list_per_sample in zip(HEX_batch_candidates_T, HEX_batch_lls_T):
            now_question_HEX_candidates = []

            for gen_text, nll in zip(cand_list_per_sample, ll_list_per_sample):
                if dataset == 'gsm8k':
                    now_extracted = gsm_extract_ans_from_generation(gen_text)
                elif dataset in ('math', 'arcc', 'truthfulqa'):
                    now_extracted = parse_math_answers(gen_text)
                else:
                    now_extracted = None

                now_question_HEX_candidates.append((now_extracted, nll))

            majority_voted_list, is_tied = most_frequent_elements_exclude_None_and_tie_lownll(
                now_question_HEX_candidates, selecting_rule='frequency'
            )
            majority_voted_list = majority_voted_list[0]
            now_question_HEX_output = majority_voted_list[0]
            HEX_outputs.append(now_question_HEX_output)

        generated_texts = HEX_outputs
        assert len(generated_texts) == len(gt_answers), f"length mismatch: gen={len(generated_texts)} gt={len(gt_answers)}"

        example_result = [
            {
                "question": questions[j],
                "prompt_input": prompts[j],
                "generations": generated_texts[j],
                "ground_truth": gt_answers[j],
            }
            for j in range(len(generated_texts))   # <- gt_answers와 동일하긴 하지만 안전하게 generated_texts 기준
        ]


        all_generations.extend(example_result)
        total_processed += len(generated_texts)
        wall_times.append(time.time() - start_time)

        # Print individual results
        if dist.get_rank() == 0:
            idx = random.randint(0, len(questions) - 1)
            print(f"Question: {questions[idx]}")
            print("-" * 50)
            print("Generation:")
            print(generated_texts[idx])
            print("-" * 50)
            print(f"Ground truth: {gt_answers[idx]}")

    avg_wall_time = sum(wall_times) / len(wall_times)
    metrics = {
        "wall_time": avg_wall_time,
        "generations": all_generations,
        "total_processed": total_processed.item(),
    }
    return metrics


class CustomDistributedSampler(DistributedSampler):
    """
    From torch docs:
    drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas

    We want drop_last = False, but don't want to have extra padding indices. Hence using a custom sampler.
    """

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        drop_last=False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
        else:
            # If we don't drop the last batch, we need to calculate the number of samples per rank.
            self.total_size = len(self.dataset)
            self.num_samples = len(self.dataset) // self.num_replicas + int(
                rank < (self.total_size % self.num_replicas)
            )

        self.shuffle = shuffle
        self.seed = seed


if __name__ == "__main__":
    import time
    start_time = time.time()
    # Note: This evaluation script saves only model generations. A separate parser is used later to extract
    # predictions and calculate metrics.

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/data1/shared/LLaDA-8B-Instruct/")
    parser.add_argument("--few_shot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--dataset", type=str, choices=[
            "gsm8k", 
            "math", 
            'arcc', 
            'truthfulqa',   
       ], default="gsm8k"
    )
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--block_lengths", nargs='+', type=int, default=[8,16,32,64,128])
    parser.add_argument("--diffusion_steps", type=int, default=128)
    parser.add_argument("--add_reasoning", action="store_true")
    parser.add_argument("--dont_save", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--dont_use_box", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    
    print("HEX block_lengths:", args.block_lengths)
    init_seed(args.seed)
    print(f'seed_num: {args.seed}')

    local_rank = setup_ddp()
    # args.diffusion_steps = args.gen_length // 2
    num_evals = {"gsm8k": -1, "math": -1, 'arcc':-1, 'truthfulqa':-1}

    # for truthfulqa, -1 only
    assert num_evals['truthfulqa'] == -1

    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=False, torch_dtype=torch.bfloat16).to(
        local_rank
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    
    
    if args.checkpoint_path != 'nope':
        model_state = 'LORA_MERGED'
        print('='*30)
        print('YOU GAVE A LORA PATH. USING MERGED MODEL !!!!!!!!!!!!')
        print(f'foundation model: {args.model_path}')
        print(f'lora adapter: {args.checkpoint_path}')
        print('='*30)
        model = PeftModel.from_pretrained(model, args.checkpoint_path, torch_dtype=torch.bfloat16).to(
            local_rank
        )

        if dist.get_world_size() > 1:
            dist.barrier()  # Make sure all processes are ready
            for param in model.parameters():
                dist.broadcast(param.data, src=0)
            print(f"Rank {local_rank}: Parameters synchronized")
    else:
        model_state = 'LORA_NONE'
        print('='*30)
        print('NO LORA PATH. YOU GAVE "nope". NOT MERGED MODEL !!!!!!!!!!!!')
        print(f'foundation model: {args.model_path}')
        print(f'lora adapter: NOT USING')
        print('='*30)


    dataset = DATASET_MAP[args.dataset](
        tokenizer,
        subsample=num_evals[args.dataset],
        num_examples=args.few_shot,
        add_reasoning=True,  # prefill for all models
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=CustomDistributedSampler(dataset, shuffle=False),
        collate_fn=dataset.collate_fn,
    )

    init_seed(args.seed)

    model_name = args.model_path.split("/")[-1]
    os.makedirs(args.output_dir, exist_ok=True)

    from datetime import datetime
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    filename = f"{args.output_dir}/{timestamp}_{args.dataset}_{model_name}_{model_state}_genlen{args.gen_length}_diffsteps{args.diffusion_steps}_HEX_blocks{args.block_lengths}_seed{args.seed}_generations.json"
  
    metrics = evaluate(
        model,
        tokenizer,
        dataloader,
        dataset=args.dataset,
        gen_length=args.gen_length,
        steps=args.diffusion_steps,
        block_lengths=args.block_lengths,
        temperature=args.temperature,
    )

    total_latency = time.time() - start_time
    if not args.dont_save:
        with open(filename, "w") as f:
            json.dump(
                {
                    "generations": metrics["generations"],
                    "metrics": {
                        "wall_time": metrics["wall_time"],
                        "total_processed": metrics["total_processed"],
                    },
                    "model_path": args.model_path,
                    "checkpoint_path": args.checkpoint_path,
                    "gen_length": args.gen_length,
                    "diffusion_steps": args.diffusion_steps,
                    "block_lengths": args.block_lengths,
                    "total_latency": total_latency,
                },
                f,
                indent=2,
            )

    cleanup_ddp()
