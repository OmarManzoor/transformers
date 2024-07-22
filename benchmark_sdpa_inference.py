import argparse

import numpy as np
import pandas as pd
import torch
import gc
from tqdm import tqdm
from transformers import AutoTokenizer, AlbertForSequenceClassification


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-batches",
        type=int,
        default=50,
        help="",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="",
    )
    parser.add_argument(
        "--seqlen",
        type=int,
        default=256,
        help="Input sequence length.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="albert/albert-base-v2",
        help="",
    )
    parser.add_argument(
        "--use-cuda",
        action="store_true",
    )
    parser.add_argument(
        "--use-half",
        action="store_true",
    )
    parser.add_argument(
        "--use-mask",
        action="store_true",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
    )
    parser.add_argument(
        "--max_token",
        type=int,
        default=100,
        help="Number of new tokens, for autoregressive models using generate.",
    )
    return parser


def get_batch(batch_size, sequence_length):
    tokens = torch.randint(high=5, size=(batch_size, sequence_length))
    mask = torch.ones((batch_size, sequence_length), )
    mask[0, 0] = 0  # real world case where we may mask
    return tokens, mask


def timing_cuda(model, num_batches, input_ids, masks):

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    # We need NOT call torch.cuda.empty_cache() here as it appears to negate the warmup.

    latencies = []
    for _ in tqdm(range(num_batches)):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()

        _ = model(input_ids, masks)
        end_event.record()
        torch.cuda.synchronize()

        latency_ms = start_event.elapsed_time(end_event)
        latencies.append(latency_ms)

    max_memory = torch.cuda.max_memory_allocated(device)

    return np.mean(latencies), max_memory


def benchmark(model, input_ids, masks, num_batches, max_token, pad_token_id):
    _ = model(input_ids, masks)
    torch.cuda.synchronize()

    total_time, max_mem = timing_cuda(model, num_batches, input_ids, masks)

    return total_time, max_mem


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    if args.sweep:
        BATCH_SIZES = [1, 2, 4]
        SEQ_LEN = [128, 265]
    else:
        BATCH_SIZES = [args.batch_size]
        SEQ_LEN = [args.seqlen]

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    autoclass = AlbertForSequenceClassification

    if args.use_cuda:
        with torch.device("cuda:0"):
            hf_model = autoclass.from_pretrained(args.model_name, torch_dtype=torch.float16 if args.use_half else None,
                                                 attn_implementation="eager")
        hf_model = hf_model.to("cuda:0")
        hf_model = hf_model.to(torch.float16)
    else:
        hf_model = autoclass.from_pretrained(args.model_name, torch_dtype=torch.float16 if args.use_half else None,
                                             attn_implementation="eager")

    output_name = "log_{}.csv".format(args.model_name.replace("/", "-"))
    output_file = open(output_name, "w")
    output_file.write(
        "num_batches, batch_size, seq_len, is cuda, is half, use mask, Per token latency eager (ms), Per token latency SDPA (ms), Speedup (%), Mem eager (MB), Mem BT (MB), Mem saved (%)\n"
    )

    all_max_mem_eager = {}
    total_eager_time = {}
    for bs in tqdm(BATCH_SIZES):
        for seq_len in tqdm(SEQ_LEN):
            print(f"-- Running: bs={bs}, seq_len={seq_len}")
            input_ids, masks = get_batch(bs, seq_len)

            if args.use_cuda:
                input_ids = input_ids.to(device)
                masks = masks.to(device)

            if args.use_mask is False and bs == 1:
                masks = None

            with torch.inference_mode():
                eager_time, max_mem_eager = benchmark(
                    hf_model,
                    input_ids,
                    masks,
                    args.num_batches,
                    args.max_token,
                    tokenizer.pad_token_id,
                )

            total_eager_time[(bs, seq_len)] = eager_time
            all_max_mem_eager[(bs, seq_len)] = max_mem_eager

    del hf_model
    gc.collect()
    total_sdpa_time = {}
    all_max_mem_sdpa = {}

    if args.use_cuda:
        with torch.device("cuda:0"):
            hf_model = autoclass.from_pretrained(args.model_name, torch_dtype=torch.float16 if args.use_half else None,
                                                 attn_implementation="sdpa")
        hf_model = hf_model.to("cuda:0")
        hf_model = hf_model.to(torch.float16)
    else:
        hf_model = autoclass.from_pretrained(args.model_name, torch_dtype=torch.float16 if args.use_half else None,
                                             attn_implementation="sdpa")

    for bs in tqdm(BATCH_SIZES):
        for seq_len in tqdm(SEQ_LEN):
            print(f"-- Running: bs={bs}, seq_len={seq_len}")
            input_ids, masks = get_batch(bs, seq_len)

            if args.use_cuda:
                input_ids = input_ids.to(device)
                masks = masks.to(device)

            if args.use_mask is False and bs == 1:
                masks = None

            with torch.inference_mode():
                # raise error if no optimized kernel is available
                with torch.backends.cuda.sdp_kernel(
                        enable_flash=True, enable_math=True, enable_mem_efficient=True
                ):
                    sdpa_time, max_mem_sdpa = benchmark(
                        hf_model,
                        input_ids,
                        masks,
                        args.num_batches,
                        args.max_token,
                        tokenizer.pad_token_id,
                    )
                total_sdpa_time[(bs, seq_len)] = sdpa_time
                all_max_mem_sdpa[(bs, seq_len)] = max_mem_sdpa

            per_token_latency_eager = total_eager_time[(bs, seq_len)] / args.max_token
            per_token_latency_sdpa = total_sdpa_time[(bs, seq_len)] / args.max_token

            max_mem_eager = all_max_mem_eager[(bs, seq_len)]
            max_mem_sdpa = all_max_mem_sdpa[(bs, seq_len)]

            speedup = (per_token_latency_eager / per_token_latency_sdpa - 1) * 100
            mem_saved = (max_mem_eager / max_mem_sdpa - 1) * 100

            max_mem_eager = max_mem_eager * 1e-6
            max_mem_sdpa = max_mem_sdpa * 1e-6

            print(f"PT eager: {per_token_latency_eager:.3f} ms, peak {max_mem_eager:.2f} MB")
            print(f"PT SDPA: {per_token_latency_sdpa:.3f} ms, peak {max_mem_sdpa:.2f} MB")

            output_file.write(
                "{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    args.num_batches,
                    bs,
                    seq_len,
                    args.use_cuda,
                    args.use_half,
                    args.use_mask,
                    f"{per_token_latency_eager:.3f}",
                    f"{per_token_latency_sdpa:.3f}",
                    f"{speedup:.3f}",
                    f"{max_mem_eager:.3f}",
                    f"{max_mem_sdpa:.3f}",
                    f"{mem_saved:.3f}",
                )
            )

    output_file.close()
    print("RESULTS:")
    df = pd.read_csv(output_name)
    print(df.to_markdown(index=False))
