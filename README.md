# TPHC

Official code for the paper "[TPHC: Improving Hyper-Connections with Tensor-Parallel
Grouped Updates](https://arxiv.org/abs/2611.23026)".

TPHC maintains `num_streams` (N) parallel residual streams. Each decoder layer reads
them through a learned `PreMapping`, computes attention and MLP on `num_groups` (G)
grouped sublayer streams, and writes back `num_updates` (U) of the residual streams
through learned `PostMapping` / doubly-stochastic `ResMapping` gates. For distributed
execution, the residual-stream dimension itself is sharded across tensor-parallel
ranks: each rank owns N/tp streams end-to-end, the mapping gate
projections run partial, and the per-layer write-back is a single reduce-scatter.

## Models and configurations

| Model dimensions | Llama | mHC | TPHC |
| --- | --- | --- | --- |
| TinyLlama (1.1B) | [Config](configs/tinyllama.json) | [Config](configs/tinyllama_mhc.json) | [Config](configs/tinyllama_tphc.json) |
| Llama-3.1 (8B) | [Config](configs/llama8b.json) | [Config](configs/llama8b_mhc.json) | [Config](configs/llama8b_tphc.json) |

Both TPHC configs start with four residual streams, four sublayer groups, and four updated streams per layer. Three fields control this behavior in the [TPHC implementation](models/modeling_llama_tphc.py).

| Config field | Role |
| --- | --- |
| `num_streams` (N) | Total residual streams carried between layers. |
| `num_groups` (G) | Groups of attention heads or feed-forward channels. `PreMapping` mixes the N streams separately for each group. |
| `num_updates` (U) | Streams updated per layer. `PostMapping` mixes group outputs for each selected stream, and the doubly stochastic `ResMapping` mixes their residuals. The other streams pass through. |

The TPHC and [mHC](models/modeling_llama_mhc.py) models are registered with Hugging Face Transformers as `tphc-llama` and `mhc-llama`, respectively.

## Installation

The reference environment uses Python 3.12, CUDA 12.4, PyTorch 2.6.0+cu124, and Transformers 4.57.0. Use these PyTorch and Transformers versions when comparing timings with the paper, since compiler and attention kernels can vary between releases.

```bash
python -m pip install -r requirements.txt
```

## Pretraining

The [training script](run_clm.sh) launches [run_clm.py](run_clm.py) with Accelerate and trains a four-stream TPHC model from scratch for one epoch on MiniPile. It uses the TinyLlama dimensions, 2,048-token sequences, bfloat16, and PyTorch scaled dot-product attention (SDPA).

```bash
bash run_clm.sh
```

## Tensor-parallel benchmarks

Both examples use four GPUs on one node, randomly initialized weights, and synthetic token IDs. With `--shard_streams`, each rank holds N/4 residual streams across the decoder layers. The ranks compute partial gate projections and combine their contributions to each sublayer write with one reduce-scatter. Omit the flag to benchmark the same model with replicated residual streams.

### Training step

Measure a forward pass, backward pass, and AdamW update with the batch size and sequence length used in Table 2 of the paper. The script reports median step time and peak allocated GPU memory on rank 0.

```bash
torchrun --nproc_per_node=4 tp_bench.py \
    --config_name configs/tinyllama_tphc.json \
    --batch_size 1 --block_size 2048 \
    --shard_streams --compile --compile_mode max-autotune
```

### Prefill and decoding

Measure prefill latency and peak memory for one 2,048-token prompt, followed by median per-token decode latency using the same KV cache. This command compiles prefill with `max-autotune-no-cudagraphs` and runs decoding in eager mode.

```bash
torchrun --nproc_per_node=4 tp_infer.py \
    --config_name configs/tinyllama_tphc.json \
    --batch_size 1 --prefill_len 2048 \
    --shard_streams --compile_prefill --compile_mode max-autotune
```

To benchmark 8 or 16 streams, add `--config_overrides num_streams=8` or `--config_overrides num_streams=16` to either command. Both settings keep G and U at four.

## Zero-shot evaluation

[run_harness.py](run_harness.py) evaluates a trained checkpoint with PyTorch SDPA on HellaSwag, OpenBookQA, WinoGrande, ARC-Challenge, ARC-Easy, and PIQA using the [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness).

```bash
python run_harness.py \
    --model_name_or_path outputs/tinyllama-tphc4 \
    --output harness/tinyllama-tphc4.json
```
