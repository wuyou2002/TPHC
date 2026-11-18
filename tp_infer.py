#!/usr/bin/env python
"""Inference latency benchmark: prefill + decode (KV-cache) for llama / mhc-llama / tphc-llama.

Measures per rank:
  - prefill: one forward over a --prefill_len prompt with use_cache=True
  - decode:  token-by-token generation with the prefill's DynamicCache,
             per-token latency (median over steps)
  - TP-correctness probe: prints the first generated token ids so a
    1-rank reference run can be compared with an N-rank TP run.

The model is placed on GPU layer-by-layer while sharding (build_tp_model):
70B-class models do not fit unsharded on one GPU.

Launch:
  torchrun --nproc_per_node=4 tp_infer.py --config_name configs/tinyllama_tphc.json   # reference
"""
import argparse
import os
import statistics
import time

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

import models  # noqa: F401
from transformers import AutoConfig, AutoModelForCausalLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_name", required=True)
    p.add_argument("--config_overrides", default=None)
    p.add_argument("--shard_pre_mapping", action="store_true",
                   help="tphc-llama only: shard the PreMapping gate projection")
    p.add_argument("--shard_streams", action="store_true",
                   help="shard the residual stream dim across TP ranks (route C: "
                        "mhc/tphc stream modes; mhc write-back variant via "
                        "MHC_LEGACY_WRITEBACK=0|1, default reviewer scheme)")
    p.add_argument("--prefill_len", type=int, default=2048)
    p.add_argument("--decode_tokens", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--warmup_prefill", type=int, default=2)
    p.add_argument("--warmup_decode", type=int, default=8)
    p.add_argument("--compile", action="store_true",
                   help="torch.compile(model.forward, mode='reduce-overhead') — enables CUDA-graph "
                        "trees; implies a StaticCache (fixed KV buffers)")
    p.add_argument("--compile_prefill", action="store_true",
                   help="additionally compile the prefill with torch.compile (mode=default, fusion "
                        "only — reduce-overhead/CUDA-graphs on prefill is a known transformers failure). "
                        "The first warmup prefill stays eager (StaticCache lazy init + mark_static_address).")
    p.add_argument("--compile_mode", default="reduce-overhead", choices=["reduce-overhead", "max-autotune"],
                   help="decode compile mode. max-autotune = Triton autotune + cudagraphs; the prefill "
                        "(via --compile_prefill) then uses max-autotune-no-cudagraphs.")
    p.add_argument("--static_cache", action="store_true",
                   help="use StaticCache WITHOUT compiling (control group: isolates the cache effect)")
    p.add_argument("--max_cache_len", type=int, default=0,
                   help="StaticCache capacity; 0 = prefill+warmup+decode+8")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def build_tp_model(config, mesh, device, shard_pre_mapping, seed=1234, shard_streams=False):
    """Build + tensor-parallelize with per-layer real construction on the GPU.

    from_config on CPU materializes the FULL model with fp32 init temporaries
    (measured: 70B -> 77 min and 256 GiB peak RSS PER PROCESS; 4 ranks exceed
    the host/container RAM and get OOM-killed). Instead: build the shell on
    the meta device, then construct each layer for real, move it to the GPU,
    shard it immediately, and drop the CPU copy. Peak host RAM = one layer.

    Weights are random (each layer seeded identically on every rank, so
    replicated params match across ranks) — this is a latency benchmark; the
    module __init__ defaults keep the structured gate inits (near-one-hot
    biases etc.), so magnitudes stay realistic.
    """
    import torch.nn as nn
    from torch.distributed.tensor import Replicate
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        parallelize_module,
    )
    from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    dt = torch.bfloat16
    tp = mesh.size()

    # replicated top-level modules, constructed for real on the GPU
    torch.manual_seed(seed)
    model.model.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dt, device=device)
    torch.nn.init.normal_(model.model.embed_tokens.weight, mean=0.0, std=0.02)
    model.model.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(device=device, dtype=dt)
    model.model.rotary_emb = LlamaRotaryEmbedding(config=config).to(device)
    model.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=dt, device=device)
    torch.nn.init.normal_(model.lm_head.weight, mean=0.0, std=0.02)
    stream_tp = None
    if tp > 1 and shard_streams and config.model_type in ("tphc-llama", "mhc-llama"):
        import torch.distributed as dist
        group = mesh.get_group()
        stream_tp = (group, dist.get_rank(group), tp)
        model.model._stream_tp = stream_tp
    if tp > 1:
        model = parallelize_module(model, mesh, {"lm_head": ColwiseParallel(output_layouts=Replicate())})

    megatron_plan = {
        "self_attn.q_proj": ColwiseParallel(),
        "self_attn.k_proj": ColwiseParallel(),
        "self_attn.v_proj": ColwiseParallel(),
        "self_attn.o_proj": RowwiseParallel(),
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(),
    }
    # sparse-update (N > U) stream plan for tphc: streams move inside the MLP
    # write's reduce-scatter; identical indices to apply_tensor_parallel's.
    tphc_plan = None
    if (tp > 1 and shard_streams and config.model_type == "tphc-llama"
            and config.num_streams != config.num_updates):
        from models.modeling_llama_tphc import (
            build_stream_shard_plan, get_update_groups, stream_shard_step_indices, _StreamShardStep,
        )
        groups = [get_update_groups(i, config.num_streams, config.num_updates)
                  for i in range(config.num_hidden_layers)]
        tphc_plan = build_stream_shard_plan(groups, config.num_streams, tp)
    for i, meta_layer in enumerate(model.model.layers):
        torch.manual_seed(seed + 1 + i)
        layer = type(meta_layer)(config, i).to(device=device, dtype=dt)
        if tp > 1:
            if config.model_type == "mhc-llama" and shard_streams:
                # mirror apply_tensor_parallel's stream branch (route C; both
                # write-back modes — MHC_LEGACY_WRITEBACK selects the legacy one)
                from models.modeling_llama_mhc import (
                    PartialRowwiseLinear, _register_partial_grad_sync,
                )
                import os
                legacy = os.environ.get("MHC_LEGACY_WRITEBACK", "0") == "1"
                layer._stream_tp = stream_tp
                layer._legacy_writeback = legacy
                plan = {
                    "self_attn.q_proj": ColwiseParallel(),
                    "self_attn.k_proj": ColwiseParallel(),
                    "self_attn.v_proj": ColwiseParallel(),
                    "mlp.gate_proj": ColwiseParallel(),
                    "mlp.up_proj": ColwiseParallel(),
                }
                if legacy:
                    plan["self_attn.o_proj"] = RowwiseParallel()
                    plan["mlp.down_proj"] = RowwiseParallel()
                parallelize_module(layer, mesh, plan)
                if not legacy:
                    layer.self_attn.o_proj = PartialRowwiseLinear(
                        layer.self_attn.o_proj, stream_tp, config.hidden_size // tp)
                    layer.mlp.down_proj = PartialRowwiseLinear(
                        layer.mlp.down_proj, stream_tp, config.intermediate_size // tp)
                for sub in ("attn", "mlp"):
                    _register_partial_grad_sync(
                        getattr(layer, f"layernorm_{sub}").weight, stream_tp[0])
                    for kind in ("pre", "post", "res"):
                        _register_partial_grad_sync(
                            getattr(layer, f"{kind}_mapping_{sub}").weight, stream_tp[0])
            elif config.model_type in ("llama", "mhc-llama"):
                parallelize_module(layer, mesh, megatron_plan)
            elif config.model_type == "tphc-llama":
                if shard_streams:
                    # route C (mirror apply_tensor_parallel's stream branch)
                    layer._stream_tp = stream_tp
                    if tphc_plan is not None:
                        layer._sh_steps = tuple(
                            _StreamShardStep(stream_shard_step_indices(layer.update_groups, st, stream_tp[1]), device)
                            for st in tphc_plan[i])
                    layer.self_attn.tp_shard(mesh, pre_sliced=True, gather_output=False)
                    layer.mlp.tp_shard(mesh, slice_input=False, gather_output=False)
                else:
                    if shard_pre_mapping:
                        layer.pre_mapping_attn.tp_shard(mesh)
                        layer.pre_mapping_mlp.tp_shard(mesh)
                    layer.self_attn.tp_shard(mesh, pre_sliced=shard_pre_mapping)
                    layer.mlp.tp_shard(mesh, slice_input=not shard_pre_mapping)
            else:
                raise ValueError(f"unsupported model_type {config.model_type}")
        model.model.layers[i] = layer
    return model


def main():
    args = parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = f"cuda:{rank}"
    torch.manual_seed(args.seed)

    config = AutoConfig.from_pretrained(args.config_name)
    if args.config_overrides:
        config.update_from_string(args.config_overrides)
    config.use_cache = True

    mesh = init_device_mesh("cuda", (world,))
    model = build_tp_model(config, mesh, device, args.shard_pre_mapping, seed=args.seed,
                           shard_streams=args.shard_streams)
    model.eval()

    # ---- optional compile / StaticCache setup ----
    use_static = args.compile or args.static_cache or args.compile_prefill
    static_cache = None
    if use_static:
        from transformers import StaticCache
        max_len = args.max_cache_len or (args.prefill_len + args.warmup_decode + args.decode_tokens + 8)
        static_cache = StaticCache(
            config=config, max_batch_size=args.batch_size, max_cache_len=max_len,
            device=device, dtype=torch.bfloat16,
        )
    if args.compile:
        # reduce-overhead = CUDA-graph trees: the hundreds of tiny per-token
        # kernels (mappings / sinkhorn / scatter) collapse into one graph
        # replay, removing the launch overhead that dominates batch-1 decode.
        # NOTE: only the DECODE step is compiled — transformers' StaticCache
        # explicitly does not support compiling the prefill with
        # reduce-overhead (lazy init + mark_static_address must run eager).
        compiled_forward = torch.compile(model.forward, mode=args.compile_mode)
    prefill_forward = None
    if args.compile_prefill:
        prefill_mode = "max-autotune-no-cudagraphs" if args.compile_mode == "max-autotune" else "default"
        prefill_forward = torch.compile(model.forward, mode=prefill_mode)  # fusion; cudagraphs on prefill is a known failure
        if not args.compile:
            # SDPA-decomposition fix (2026-11-02): under torch.compile, the
            # StaticCache prefill mask (kv=2072 != q=2048, and
            # _ignore_causal_mask_sdpa refuses to fold under dynamo) reaches
            # SDPA as a materialized 4D bool mask, and the composite-
            # implicit-autograd decomposition lowers it to an explicit
            # matmul+softmax chain — materializing (b,h,s,s) score buffers
            # (4x256 MiB at 1-GPU s=2048 in vanilla's single big region;
            # traced/verified via memory snapshot). Forcing SDPA to run
            # EAGERLY graph-breaks around it and the dispatcher picks the
            # fused mem-efficient kernel instead: vanilla 1-GPU prefill peak
            # 3.32 -> 2.27 GiB (matches eager), time +-2%. Scoped to
            # 1-GPU / 70B-class (--build_layerwise): at 1.1B TP4 the 22 breaks
            # serialize the compiled fragments' NCCL pipelining and the run
            # becomes host-bound (vanilla 25.1 -> 48.6 ms) — NOT applied there
            # (also not with --compile: breaks would disable the decode
            # CUDA-graph capture). SDPA_DECOMP=1 restores the old behavior.
            if (os.environ.get("SDPA_DECOMP") != "1"
                    and not args.compile
                    # 70B-class proxy: per-layer GPU time (layers >= 40) dwarfs
                    # the ~20 ms of fragment-switch overhead the breaks add
                    and (world == 1 or config.num_hidden_layers >= 40)):
                import torch._dynamo as _dynamo
                _dynamo.disallow_in_graph(
                    torch.nn.functional.scaled_dot_product_attention)

    g = torch.Generator(device="cpu").manual_seed(args.seed)
    prompt = torch.randint(0, config.vocab_size, (args.batch_size, args.prefill_len), generator=g).cuda()

    def prefill_once(eager=False):
        if static_cache is not None:
            # explicit positions: a fresh StaticCache reports len 0, but after
            # the first prefill get_seq_length() = prefill_len and the implicit
            # arange(prefill_len, 2*prefill_len) would write past max_cache_len
            # (device-side assert). Rewriting 0..prefill_len is idempotent.
            pos = torch.arange(0, args.prefill_len, device=device, dtype=torch.long)
            fwd = model.forward
            if not eager and prefill_forward is not None:
                fwd = prefill_forward
            out = fwd(prompt, past_key_values=static_cache, cache_position=pos, use_cache=True)
        else:
            out = model(prompt, use_cache=True)
        torch.cuda.synchronize()
        return out

    def decode_step(token, cache, pos):
        if static_cache is not None:
            cache_position = torch.tensor([pos], device=device, dtype=torch.long)
            fwd = compiled_forward if args.compile else model.forward
            out = fwd(token, past_key_values=cache, cache_position=cache_position, use_cache=True)
        else:
            out = model(token, past_key_values=cache, use_cache=True)
        torch.cuda.synchronize()
        return out

    # NOTE: torch.inference_mode() is incompatible with tphc's manual
    # DTensor.from_local + bmm path (raises "Cannot set version_counter for
    # inference tensor"); no_grad works for both variants.
    with torch.no_grad():
        # ---- prefill (fresh cache per call) ----
        # first warmup stays EAGER when --compile_prefill: StaticCache lazy
        # init + mark_static_address must run outside dynamo (compiling the
        # very first call is the known-failure path)
        for w in range(args.warmup_prefill):
            prefill_once(eager=(w == 0 and prefill_forward is not None))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = prefill_once()
        prefill_ms = (time.perf_counter() - t0) * 1e3
        prefill_peak = torch.cuda.max_memory_allocated() / 2**30  # prefill-only (reset just before t0)

        cache = static_cache if static_cache is not None else out.past_key_values
        token = out.logits[:, -1:].argmax(dim=-1)
        first_tokens = token[0].tolist()

        # ---- decode (token by token, growing the prefill cache) ----
        step_ms = []
        n_total = args.warmup_decode + args.decode_tokens
        for i in range(n_total):
            token_in = token.detach()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = decode_step(token_in, cache, args.prefill_len + i)
            dt = (time.perf_counter() - t0) * 1e3
            # NOTE: under reduce-overhead cudagraphs, out.logits is a reused
            # buffer — argmax must be consumed before the next replay (it is).
            token = out.logits[:, -1:].argmax(dim=-1)
            if i < args.warmup_decode:
                first_tokens.append(token[0].item())
            else:
                step_ms.append(dt)
                if len(first_tokens) < 10:
                    first_tokens.append(token[0].item())

    med = statistics.median(step_ms)
    peak = torch.cuda.max_memory_allocated() / 2**30
    if rank == 0:
        print(f"RESULT: model={config.model_type} tp={world} batch={args.batch_size} "
              f"{'shard_pre_mapping ' if args.shard_pre_mapping else ''}"
              f"{'compiled ' if args.compile else ('static_cache ' if args.static_cache else '')}"
              f"{'(' + args.compile_mode + ')' if args.compile and args.compile_mode != 'reduce-overhead' else ''}"
              f"{'prefill_compiled ' if args.compile_prefill else ''}bf16")
        print(f"  prefill  : {args.prefill_len} tokens in {prefill_ms:.1f} ms "
              f"({args.batch_size * args.prefill_len / prefill_ms * 1e3:,.0f} tok/s)")
        print(f"  prefill peak: {prefill_peak:.2f} GiB")
        print(f"  decode   : median {med:.2f} ms/token ({1e3 / med:,.1f} tok/s), "
              f"n={len(step_ms)} after {args.warmup_decode} warmup")
        print(f"  peak mem : {peak:.2f} GiB")
        print(f"  first tokens: {first_tokens[:10]}")
    if args.compile:
        # NCCL collectives captured inside the decode CUDA graphs make later
        # EAGER collectives on the same PG hang (the final barrier deadlocks
        # and the NCCL watchdog SIGABRTs after 480s). Flush and hard-exit —
        # bench-only; results are already printed above.
        torch.cuda.synchronize()
        os._exit(0)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
