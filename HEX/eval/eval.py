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



from generate import generate
from generate import generate_llada
from generate import generate_with_dual_cache
from generate import generate_with_prefix_cache
from generate import generate_DynamicBlock
from generate import generate_block_search


from gsm8k import GSM8KDataset
from math500 import MATH500Dataset
from countdown import CTDDataset
from sudoku import SudokuDataset

from arcc import ARCCDataset
from truthfulqa import TruthfulQAMCDataset


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
    gen_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    steps=64,
    block_length=32,
    decoding="",
    kv_caching="baseline", 
    early_ban_steps=10,
    dynamic_block_sizes_list=[],
    block_search_candidates=[],
    save_intermediate = False,
    ############
    # "baseline"
    # "prefix_cache"
    # "parallel"
    # "parallel_factor"
    # "prefix_cache_parallel"
    # "dual_cache_parallel"
    # "prefix_cache_parallel_factor"
    # "dual_cache_parallel_factor"
    ############
):
    model.eval()
    total_processed = torch.tensor(0, device=model.device)
    wall_times = []
    all_generations = []
    device = model.device

    for batch in tqdm(dataloader, disable=(dist.get_rank() != 0)):
        start_time = time.time()
        input_ids = batch["input_ids"].to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]
        
        if kv_caching != 'baseline':
            print(f'Using kv_caching: {kv_caching}')

        ############################################
        # zero shot GSM8K
        ############################################
        # baseline latency: 6.180189847946167
        # ------------------------------
        # prefix cache latency: 5.244270086288452
        # ------------------------------
        # parallel latency: 3.4129531383514404
        # ------------------------------
        # parallel factor latency: 2.4383838176727295
        # ------------------------------
        # prefix cache+parallel latency: 4.0147459506988525
        # ------------------------------
        # dual cache+parallel latency: 4.003870487213135
        # ------------------------------
        # prefix cache+parallel factor latency: 3.2430648803710938
        # ------------------------------
        # dual cache+parallel factor latency: 3.2073497772216797
        ############################################

        if decoding == "low_confidence_banAEoTearly" and kv_caching != "baseline":
            raise ValueError(f"banAEoTearly is only for 'baseline' now")
        
        
        if kv_caching == "baseline" : 
            if save_intermediate == False: 
                # small json file: output, ll
                out, ll = generate(model, input_ids, tokenizer, steps=steps, gen_length=gen_length, block_length=block_length, temperature=temperature, cfg_scale=cfg_scale, 
        remasking=decoding, early_ban_steps=early_ban_steps, save_intermediate=False)
            else: 
                # big json file: output, ll, intermediate_steps, confidences
                out, ll, intermdeiates, confidences = generate(model, input_ids, tokenizer, steps=steps, gen_length=gen_length, block_length=block_length, temperature=temperature, cfg_scale=cfg_scale, 
        remasking=decoding, early_ban_steps=early_ban_steps, save_intermediate=True)
       
       
        # intermediate output not set for other kv_caching yet! 20250810
        elif kv_caching == "prefix_cache" : out, nfe = generate_with_prefix_cache(model, input_ids, tokenizer, steps=steps, gen_length=gen_length, block_length=block_length, temperature=temperature, cfg_scale=cfg_scale, remasking=decoding)
        elif kv_caching == "parallel" : out, nfe = generate(model, input_ids, tokenizer, steps=steps, gen_length=gen_length, block_length=block_length, temperature=temperature, remasking=decoding, threshold=0.9)
        elif kv_caching == "parallel_factor" : out, nfe = generate(model, input_ids, tokenizer, steps=steps, gen_length=gen_length, block_length=block_length, temperature=temperature, remasking=decoding, factor=1.0)
        elif kv_caching == "prefix_cache_parallel" : out, nfe = generate_with_prefix_cache(model, input_ids, tokenizer, steps=steps, gen_length=gen_length, block_length=block_length, temperature=temperature, remasking=decoding, threshold=0.9)
        elif kv_caching == "dual_cache_parallel" : out, nfe = generate_with_dual_cache(model, input_ids, tokenizer, steps=steps, gen_length=gen_length, block_length=block_length, temperature=temperature, remasking=decoding, threshold=0.9)
        elif kv_caching == "prefix_cache_parallel_factor" : out, nfe = generate_with_prefix_cache(model, input_ids, tokenizer, steps=steps, gen_length=gen_length, block_length=block_length, temperature=temperature, remasking=decoding, factor=1.0)
        elif kv_caching == "dual_cache_parallel_factor" : out, nfe = generate_with_dual_cache(model, input_ids, tokenizer, steps=steps, gen_length=gen_length, block_length=block_length, temperature=temperature, remasking=decoding, factor=1.0)
        # intermediate output not set for other kv_caching yet! 20250810


        elif kv_caching == "dynamic_blocks" : out, ll = generate_DynamicBlock(
            model, input_ids, tokenizer, steps=steps, gen_length=gen_length, dynamic_block_sizes_list=dynamic_block_sizes_list, temperature=temperature, cfg_scale=cfg_scale, 
        remasking=decoding, early_ban_steps=early_ban_steps)


        elif kv_caching == 'block_search': out, ll = generate_block_search(
            model, input_ids, tokenizer, steps=steps, gen_length=gen_length, candidate_block_size_list=block_search_candidates, temperature=temperature, cfg_scale=cfg_scale,
        remasking=decoding, early_ban_steps=early_ban_steps)



        generated_texts = tokenizer.batch_decode(out[:, -gen_length:], skip_special_tokens=False)
        # intermediate_output_strings = nfe[0] #strings # not set yet for other decodings than baseline!! 20250810 OOM...
        # intermediate_output_tokentypes = nfe #types, T, M, A, E for TextToken, MaskToken, AfterEOTToken, EOTToken


        if not save_intermediate:
            example_result = [
                {
                    "question": questions[j],
                    "prompt_input": prompts[j],
                    "generations": generated_texts[j],
                    "ground_truth": gt_answers[j],
                    # "intermediate_str": intermediate_output_strings, # OOM
                    # "intermediate_types": intermdeiates[j],
                    # "confidence_info": top_confidence_info[j],
                    "loglikelihood": ll[j],
                    # "loglikelihood_noAfterEoT": ll_noAfterEoT[j]
                }
                for j in range(len(gt_answers))
            ]

        else:
            example_result = [
                {
                    "question": questions[j],
                    "prompt_input": prompts[j],
                    "generations": generated_texts[j],
                    "ground_truth": gt_answers[j],
                    # "intermediate_str": intermediate_output_strings, # OOM
                    "intermediate_types": intermdeiates[j],
                    # "confidence_info": top_confidence_info[j],
                    "loglikelihood": ll[j],
                    # "loglikelihood_noAfterEoT": ll_noAfterEoT[j]
                }
                for j in range(len(gt_answers))
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
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--diffusion_steps", type=int, default=128)
    parser.add_argument("--add_reasoning", action="store_true")
    parser.add_argument("--dont_save", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--dont_use_box", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decoding", default="ddd")
    parser.add_argument("--kv_caching", default="baseline")
    parser.add_argument("--early_ban_steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dynamic_block_sizes_list", nargs='+', type=int, default=[])
    parser.add_argument("--block_search_candidates", nargs='+', type=int, default=[])
    args = parser.parse_args()

    if args.kv_caching == "dynamic_blocks":
        print("Dynamic block Parsed sizes:", args.dynamic_block_sizes_list)
    elif args.kv_caching == 'block_search':
        print("Block search Parsed sizes:", args.block_search_candidates)
   
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
        print('='*30)
        print('YOU GAVE A LORA PATH. USING MERGED MODEL !!!!!!!!!!!!')
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
        print('='*30)
        print('NO LORA PATH. YOU GAVE "nope". NOT MERGED MODEL !!!!!!!!!!!!')
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

    # if len(args.checkpoint_path):
    #     model_name = args.checkpoint_path.split("/")
    #     model_name = model_name[-2] + "_" + model_name[-1]
    # else:
    #     model_name = "instruct" if "Instruct" in args.model_path else "base"

    # if args.few_shot > 0:
    #     model_name = model_name + f"_fs{args.few_shot}"

    # if len(args.suffix) > 0:
    #     model_name = model_name + f"_{args.suffix}"


    
    init_seed(args.seed)


    model_name = args.model_path.split("/")[-1]
    os.makedirs(args.output_dir, exist_ok=True)

    from datetime import datetime
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    filename = f"{args.output_dir}/{timestamp}_{args.dataset}_{model_name}_{args.decoding}_genlen{args.gen_length}_diffsteps{args.diffusion_steps}_block{args.block_length}_seed{args.seed}_kv{args.kv_caching}_generations.json"
    if args.dynamic_block_sizes_list:
        filename = f"{args.output_dir}/{timestamp}_{args.dataset}_{model_name}_{args.decoding}_genlen{args.gen_length}_diffsteps{args.diffusion_steps}_blockinfo{args.dynamic_block_sizes_list}_seed{args.seed}_generations.json"

    metrics = evaluate(
        model,
        tokenizer,
        dataloader,
        gen_length=args.gen_length,
        block_length=args.block_length,
        steps=args.diffusion_steps,
        decoding=args.decoding,
        kv_caching=args.kv_caching,
        early_ban_steps=args.early_ban_steps,
        temperature=args.temperature,
        dynamic_block_sizes_list=args.dynamic_block_sizes_list,
        block_search_candidates=args.block_search_candidates
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
                    "block_length": args.block_length,
                    "total_latency": total_latency,
                    "early_ban_steps": args.early_ban_steps,
                },
                f,
                indent=2,
            )

    cleanup_ddp()
