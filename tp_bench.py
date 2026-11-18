#!/usr/bin/env python
"""TP4 training-efficiency benchmark: tphc-llama vs mhc-llama.

No checkpoints, no lr schedule — just: same data, same batch, N steps,
report median s/step, tokens/s, and peak memory per rank.

Launch: torchrun --nproc_per_node=4 tp_bench.py --config_name configs/tinyllama_tphc.json
"""
import argparse
import os
import statistics
import time

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

import models  # noqa: F401
from transformers import AutoConfig, AutoModelForCausalLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_name", required=True)
    p.add_argument("--config_overrides", default=None)
    p.add_argument("--num_steps", type=int, default=30)
    p.add_argument("--warmup_steps", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--block_size", type=int, default=2048)
    p.add_argument("--shard_pre_mapping", action="store_true",
                   help="tphc-llama only: shard the PreMapping gate projection "
                        "along its output dim so each rank computes only its local fractions")
    p.add_argument("--shard_streams", action="store_true",
                   help="tphc-llama only (route C): shard the residual stream dim across "
                        "TP ranks — mapping gates partial + reduce-scatter read/write, "
                        "o_proj/down stop all-gathering. Dense N==U regime only.")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile(model.forward) with mode=default: aot_autograd compiles "
                        "fwd AND bwd (fusion of the elementwise-heavy mapping/backward chains). "
                        "No cudagraphs (mode=reduce-overhead is unsafe for training + NCCL).")
    p.add_argument("--compile_mode", default="default", choices=["default", "max-autotune"],
                   help="torch.compile mode. max-autotune: Triton GEMM templates + epilogue fusion "
                        "(e.g. the res-mix bmm + add), at a much longer compile time.")
    p.add_argument("--build_layerwise", action="store_true",
                   help="build via tp_infer.build_tp_model (meta shell + per-layer GPU "
                        "construction): required for 70B-class models that don't fit "
                        "unsharded on one GPU. Weights become per-layer-seeded random "
                        "(latency-equivalent; see tp_infer.build_tp_model).")
    p.add_argument("--forward_only", action="store_true",
                   help="measure forward only (no labels/CE, no_grad, no optimizer) — a prefill "
                        "proxy that CAN use max-autotune WITH cudagraphs, unlike tp_infer's "
                        "prefill which StaticCache forces to no-cudagraphs. Use --batch_size 1 "
                        "to match tp_infer's prefill workload.")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def main():
    args = parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = f"cuda:{rank}"
    torch.manual_seed(args.seed)

    config = AutoConfig.from_pretrained(args.config_name)
    config.use_cache = False
    if args.config_overrides:
        config.update_from_string(args.config_overrides)

    if config.model_type == "mhc-llama":
        from models.modeling_llama_mhc import apply_tensor_parallel
        tp_kwargs = {"shard_streams": args.shard_streams}
    elif config.model_type == "tphc-llama":
        from models.modeling_llama_tphc import apply_tensor_parallel
        tp_kwargs = {"shard_pre_mapping": args.shard_pre_mapping,
                     "shard_streams": args.shard_streams}
    elif config.model_type == "llama":
        # vanilla Llama: standard Megatron plan (same as tp_infer.build_tp_model)
        def apply_tensor_parallel(model, mesh):
            from torch.distributed.tensor import Replicate
            from torch.distributed.tensor.parallel import (
                ColwiseParallel, RowwiseParallel, parallelize_module,
            )
            if mesh.size() == 1:
                # match mhc/tphc (plain model at tp==1): running the Megatron
                # DTensor plan on a 1-rank mesh adds ~9 GiB of DTensor autograd
                # overhead in the compiled graph (8B 1-GPU: 63.66 vs 54.72 GiB,
                # traced 2026-11-03) — the pre-2026-11-03 1-GPU vanilla rows
                # were measured WITH that wrap and are not plain-vanilla.
                return model
            return parallelize_module(model, mesh, {
                "model.layers.*.self_attn.q_proj": ColwiseParallel(),
                "model.layers.*.self_attn.k_proj": ColwiseParallel(),
                "model.layers.*.self_attn.v_proj": ColwiseParallel(),
                "model.layers.*.self_attn.o_proj": RowwiseParallel(),
                "model.layers.*.mlp.gate_proj": ColwiseParallel(),
                "model.layers.*.mlp.up_proj": ColwiseParallel(),
                "model.layers.*.mlp.down_proj": RowwiseParallel(),
                "lm_head": ColwiseParallel(output_layouts=Replicate()),
            })
        tp_kwargs = {}
    else:
        raise ValueError(f"unsupported model_type {config.model_type}")

    if args.build_layerwise:
        # 70B-class: from_config().to(device) needs the FULL model on every GPU
        # (~138 GiB) and OOMs — build layer-by-layer with sharding instead.
        from tp_infer import build_tp_model
        mesh = init_device_mesh("cuda", (world,))
        model = build_tp_model(config, mesh, device, args.shard_pre_mapping, seed=args.seed,
                               shard_streams=args.shard_streams)
    else:
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16).to(device)
        mesh = init_device_mesh("cuda", (world,))
        model = apply_tensor_parallel(model, mesh, **tp_kwargs)
    model.train() if not args.forward_only else model.eval()
    if args.compile:
        # mode=default (no cudagraphs): aot_autograd compiles the backward too.
        # tphc's custom autograd Functions (_TpSliceStreams etc.) cause graph
        # breaks around themselves — the regions in between still fuse.
        model.forward = torch.compile(model.forward, mode=args.compile_mode)

    local_params = sum((p.to_local() if isinstance(p, DTensor) else p).numel() for p in model.parameters())
    if rank == 0:
        print(f"model={config.model_type} | local params={local_params/1e6:.1f}M | tp={world}")

    optimizer = None if args.forward_only else torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)

    # synthetic fixed-length data (identical on both TP ranks; content is
    # irrelevant for throughput)
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    data = torch.randint(0, config.vocab_size, (256, args.block_size), generator=g).cuda()

    step_times = []
    for step in range(args.warmup_steps + args.num_steps):
        batch = data[(step * args.batch_size) % 256: (step * args.batch_size) % 256 + args.batch_size]
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        if args.forward_only:
            with torch.no_grad():
                out = model(batch)  # no labels: skip the CE fp32-logits path
        else:
            out = model(batch, labels=batch.clone())
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        if step >= args.warmup_steps:
            step_times.append(time.perf_counter() - t0)

    med = statistics.median(step_times)
    tok_s = args.batch_size * args.block_size * world / med  # pure TP: world ranks share one batch
    # tokens/s above counts the logical batch once; per-GPU memory below
    peak = torch.cuda.max_memory_allocated() / 2**30
    if rank == 0:
        print(f"RESULT: model={config.model_type} tp={world} batch={args.batch_size}x{args.block_size} "
              f"{'shard_pre_mapping ' if args.shard_pre_mapping else ''}"
              f"{'shard_streams ' if args.shard_streams else ''}"
              f"{'compiled' if args.compile else 'eager'}"
              f"{'(' + args.compile_mode + ')' if args.compile and args.compile_mode != 'default' else ''}")
        print(f"  median s/step = {med*1000:.1f} ms")
        print(f"  tokens/s      = {tok_s:,.0f}")
        print(f"  peak mem/rank = {peak:.2f} GiB")
        if args.forward_only:
            print(f"  logits mean   = {out.logits.float().mean().item():.4f}")
        else:
            print(f"  loss (last)   = {out.loss.item():.4f}")
    if args.compile:
        # compiled regions + NCCL occasionally make the teardown barrier hang
        # (watchdog SIGABRT after 480s; seen with max-autotune). Results are
        # printed above — flush and hard-exit (bench-only).
        torch.cuda.synchronize()
        os._exit(0)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
