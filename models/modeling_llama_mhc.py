from typing import List, Optional, Union

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from transformers.utils.generic import check_model_inputs
from transformers.models.llama.modeling_llama import (
    LlamaMLP,
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaForCausalLM,
    LlamaRMSNorm
)
from .configuration_llama_mhc import MyLlamaConfig

# Route-C stream-sharding primitives, shared with (and debugged in) tphc:
from .modeling_llama_tphc import (
    _TpAllReduce,
    _TpReduceScatterStreams,
    _TpSumGrad,
    _TpExpandEntry,
    _TpCollapseExit,
    _register_partial_grad_sync,
)


logger = logging.get_logger(f"transformers.{__name__}")


def sinkhorn_knopps(log_alpha, iters = 20):
    # Ablation switch (kernel-cost measurement ONLY): MHC_SKIP_SINKHORN=1
    # bypasses the 20-iteration doubly-stochastic normalization. Without it
    # the mixing matrix is NOT row/col-normalized, so outputs and the loss
    # are not a valid model — use only for speed ablations.
    if os.environ.get("MHC_SKIP_SINKHORN", "0") == "1":
        return log_alpha
    dtype = log_alpha.dtype
    log_alpha = log_alpha.float()

    log_alpha = log_alpha - log_alpha.amax(dim = -2, keepdim = True).detach()

    alpha = log_alpha.exp()

    for _ in range(iters):
        alpha = F.normalize(alpha, p = 1, dim = -2)
        alpha = F.normalize(alpha, p = 1, dim = -1)

    return alpha.to(dtype)


class TemporalFeatureAugmentation(nn.Module):
    """Causal depthwise temporal branches plus per-token Gram-Schmidt."""

    def __init__(
        self,
        hidden_size: int,
        kernel_sizes: List[int],
        gram_schmidt: bool,
        initializer_range: float,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_sizes = list(kernel_sizes)
        self.gram_schmidt = gram_schmidt
        self.eps = 1e-12
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    kernel_size=kernel_size,
                    groups=hidden_size,
                    bias=False,
                    padding=0,
                )
                for kernel_size in self.kernel_sizes
            ]
        )
        for conv in self.convs:
            nn.init.zeros_(conv.weight)
            conv._is_hf_initialized = True

    @property
    def rank(self) -> int:
        return 1 + len(self.convs)

    def _causal_conv(self, x: torch.Tensor, conv: nn.Conv1d, kernel_size: int) -> torch.Tensor:
        # [B,S,C] -> zero-pad left -> [B,C,S+K-1] -> DWConv -> [B,S,C]
        required = kernel_size - 1
        prefix = x.new_zeros(x.shape[0], required, x.shape[-1])
        conv_input = torch.cat((prefix, x), dim=1).transpose(1, 2)
        # Conv1d is cross-correlation. Flip the kernel so weight[i] has the
        # exact lag-i meaning: sum_i weight[i] * out[t-i].
        convolved = F.conv1d(
            conv_input,
            conv.weight.flip(-1),
            bias=None,
            stride=1,
            padding=0,
            groups=self.hidden_size,
        )
        return convolved.transpose(1, 2)

    def _orthogonalize(self, raw: torch.Tensor) -> torch.Tensor:
        if not self.gram_schmidt or raw.shape[-2] == 1:
            return raw
        basis: List[torch.Tensor] = [raw[..., 0, :].float()]
        for component_index in range(1, raw.shape[-2]):
            source = raw[..., component_index, :].float()
            vector = source
            for previous in basis:
                numerator = (source * previous).sum(dim=-1, keepdim=True)
                denominator = previous.square().sum(dim=-1, keepdim=True).clamp_min(self.eps)
                vector = vector - (numerator / denominator) * previous
            basis.append(vector)
        return torch.stack(basis, dim=-2).to(raw.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = [x]
        for conv, kernel_size in zip(self.convs, self.kernel_sizes):
            branches.append(self._causal_conv(x, conv, kernel_size))
        raw = torch.stack(branches, dim=-2)  # [B,S,Kr,C]
        return self._orthogonalize(raw)


class PreMapping(nn.Module):
    def __init__(self, hidden_size: int, num_streams: int, k:int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size * num_streams, num_streams))
        bias = torch.full((num_streams,), -3.0)
        bias[k] = 3.0
        self.bias = nn.Parameter(bias)
        self.alpha = nn.Parameter(torch.ones(1) * 0.01)

    def forward(self, x: torch.Tensor, hidden_states: torch.Tensor):
        h = self.alpha * torch.einsum("bld,dn->bln", x, self.weight) + self.bias
        h = F.sigmoid(h)
        return torch.einsum("bln,blnd->bld", h, hidden_states)

class PostMapping(nn.Module):
    def __init__(self, hidden_size: int, num_streams: int, temporal_rank: int = 1) -> None:
        super().__init__()
        self.num_streams = num_streams
        self.temporal_rank = temporal_rank
        self.weight = nn.Parameter(torch.zeros(hidden_size * num_streams, num_streams * temporal_rank))
        bias = torch.zeros(num_streams, temporal_rank)
        self.bias = nn.Parameter(bias)
        self.alpha = nn.Parameter(torch.ones(1) * 0.01)

    def forward(self, x: torch.Tensor, hidden_states: torch.Tensor):
        bsz, q_len, _ = x.size()
        h = self.alpha * torch.einsum("bld,dn->bln", x, self.weight).reshape(
            bsz, q_len, self.num_streams, self.temporal_rank) + self.bias
        h = 2.0 * F.sigmoid(h)
        if self.temporal_rank == 1:
            return torch.einsum("bln,bld->blnd", h.squeeze(-1), hidden_states)
        return torch.einsum("blmn,blnd->blmd", h, hidden_states)

class ResMapping(nn.Module):
    def __init__(self, hidden_size: int, num_streams: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size * num_streams, num_streams * num_streams))
        self.bias = nn.Parameter(torch.full((num_streams, num_streams), -3.0) + 6.0 * torch.eye(num_streams))
        self.alpha = nn.Parameter(torch.ones(1) * 0.01)
        self.num_streams = num_streams

    def forward(self, x: torch.Tensor, hidden_states: torch.Tensor):
        bsz, q_len, _ = x.size()
        h = self.alpha * torch.einsum("bld,dn->bln", x, self.weight).reshape(
            bsz, q_len, self.num_streams, -1) + self.bias
        h = sinkhorn_knopps(h)
        return torch.einsum("blmn,blnd->blmd", h, hidden_states)

class PartialRowwiseLinear(nn.Module):
    """Rowwise (input-dim) weight sharding WITHOUT the forward all-reduce,
    with the PARAM itself stored as the local column block.

    In stream-shard mode the sublayer output feeds the fused post/res
    reduce-scatter, which sums the per-rank partials anyway — the Megatron
    o_proj/down all-reduce is redundant (Σ_p h_post·o_p = h_post·Σ_p o_p,
    the replicated post gates commute with the collective). The partial
    sublayer output q_p is broadcast by the post gates to all N streams and
    summed inside the write-back RS.

    Each rank stores ONLY its column block W[:, lo:hi] as the Parameter
    (snapshot-diff finding 2026-11-02: holding the FULL replicated weight
    cost +1.44 GiB/rank in param+grad+optimizer footprint vs the old
    RowwiseParallel scheme). Sound because W[:, blk] is touched ONLY by its
    owner — its owner's x_local (this rank's attention/MLP output) and the
    full d(o) delivered by the write-back RS backward — so the block grad
    dW = d(o)^T @ x_local is COMPLETE locally: no replication, no backward
    collective, param/grad/AdamW states ÷ tp. State-dict shape differs from
    single-rank (like any TP shard); bias must be None (a per-rank bias add
    would be summed tp times)."""

    def __init__(self, linear, stream_tp, in_l):
        super().__init__()
        assert linear.bias is None, "PartialRowwiseLinear requires bias-free linears"
        rank = stream_tp[1]
        self.weight = nn.Parameter(
            linear.weight.data[:, rank * in_l:(rank + 1) * in_l].contiguous(),
            requires_grad=linear.weight.requires_grad,
        )

    def forward(self, x):
        from torch.distributed.tensor import DTensor
        if isinstance(x, DTensor):
            x = x.to_local()
        return F.linear(x, self.weight)


class MyLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: MyLlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.config = config

        self.temporal = (
            TemporalFeatureAugmentation(
                hidden_size=config.hidden_size,
                kernel_sizes=config.temporal_kernel_sizes,
                gram_schmidt=config.gram_schmidt,
                initializer_range=config.initializer_range,
            )
            if config.temporal_aug
            else None
        )
        temporal_rank = self.temporal.rank if self.temporal is not None else 1
        self.pre_mapping_attn = PreMapping(config.hidden_size, config.num_streams, layer_idx % config.num_streams)
        self.post_mapping_attn = PostMapping(config.hidden_size, config.num_streams)
        self.res_mapping_attn = ResMapping(config.hidden_size, config.num_streams)
        self.pre_mapping_mlp = PreMapping(config.hidden_size, config.num_streams, layer_idx % config.num_streams)
        self.post_mapping_mlp = PostMapping(config.hidden_size, config.num_streams, temporal_rank)
        self.res_mapping_mlp = ResMapping(config.hidden_size, config.num_streams)
        self.layernorm_attn = LlamaRMSNorm(config.hidden_size * config.num_streams, eps=config.rms_norm_eps)
        self.layernorm_mlp = LlamaRMSNorm(config.hidden_size * config.num_streams, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx
        # Route-C stream sharding (set by apply_tensor_parallel): (group, rank, tp)
        self._stream_tp = None
        # MHC_LEGACY_WRITEBACK=1: pre-reviewer write-back (o_proj/down keep the
        # Megatron rowwise AR; post writes only the local m-block) — 6C/sublayer
        # instead of the reviewer scheme's 4.5C, but smaller write-back partials.
        self._legacy_writeback = False

    # ---------------- route C: stream-sharded forward (semantics preserved) ----------------

    def _sh_norm_gates(self, norm, x_local, pre, post, res):
        """Wide-norm statistics + all three mhc mapping gate projections in ONE
        all-reduce (ported from tphc's _sh_norm_gates: was a separate (b, s)
        stats AR + a logits AR — one fewer collective and graph break per
        sublayer).

        Linearity makes the fusion exact: the gates consume
        einsum((x * inv * w), W) == inv * einsum((x * w), W), so the partial
        logits are computed from the UN-normalized local block with the norm
        weight FOLDED INTO the matmul weight (Wf = W * w[:, None] — the
        full-width activation multiply becomes a (K_l, cols) weight
        multiply), and the norm scale `inv` is applied after the single
        all-reduce that also carries the sum-of-squares (computed as bf16
        squares + fp32 sum(dtype=) accumulation). The norm OUTPUT is never
        materialized (only _sh_gates consumed it). mhc gate formulas:
        pre = sigmoid, post = 2*sigmoid (temporal_rank Kr), res = sinkhorn.

        tier-1: the weight slices are PLAIN narrows (dynamo-traceable — no
        custom-Function graph break, so this projection fuses with the
        surrounding region); their block grads are reassembled by
        post-accumulate-grad all-reduce hooks registered at shard setup
        (see _register_partial_grad_sync), running outside the compiled
        regions."""
        bsz, q_len, _ = x_local.shape
        group, rank, tp = self._stream_tp
        N = self.config.num_streams
        Kr = post.temporal_rank
        K_l = x_local.shape[-1]
        cols = N + N * Kr + N * N
        lo, hi = rank * K_l, (rank + 1) * K_l
        w = norm.weight[lo:hi]                                             # (K_l,)
        W = torch.cat([pre.weight, post.weight, res.weight], dim=1)[lo:hi, :]
        Wf = W * w.unsqueeze(-1)                                           # fold the norm weight in
        partial_raw = torch.einsum("bld,dn->bln", x_local, Wf).float()     # (b, s, cols)
        sumsq = x_local.square().sum(dim=-1, dtype=torch.float32)          # (b, s)
        merged = _TpAllReduce.apply(
            torch.cat([partial_raw, sumsq.unsqueeze(-1)], dim=-1), group)
        raw_u, ss = merged.split([cols, 1], dim=-1)
        # local consumers of the shared stats -> partial grads must be summed
        ss = _TpSumGrad.apply(ss, group)
        inv = torch.rsqrt(ss.squeeze(-1) / norm.weight.shape[0] + norm.variance_epsilon)
        raw = (raw_u * inv.unsqueeze(-1)).to(x_local.dtype)
        raw_pre, raw_post, raw_res = raw.split([N, N * Kr, N * N], dim=-1)
        g_pre = torch.sigmoid(pre.alpha * raw_pre + pre.bias)
        g_post = 2.0 * torch.sigmoid(post.alpha * raw_post.view(bsz, q_len, N, Kr) + post.bias)
        g_res = sinkhorn_knopps(res.alpha * raw_res.view(bsz, q_len, N, N) + res.bias)
        gates = _TpSumGrad.apply(torch.cat([
            g_pre.reshape(bsz, q_len, -1), g_post.reshape(bsz, q_len, -1),
            g_res.reshape(bsz, q_len, -1)], dim=-1), group)
        gp, go, gr = gates.split([N, N * Kr, N * N], dim=-1)
        return (gp.view(bsz, q_len, N), go.view(bsz, q_len, N, Kr), gr.view(bsz, q_len, N, N))

    def _sh_pre_read(self, g_pre, hs_local):
        """Sharded PreMapping collapse: partial sum over the local streams,
        then ALL-REDUCE — the single (b, s, h) sublayer input must be
        REPLICATED (Megatron colwise consumes it on every rank). Backward
        through _TpAllReduce is identity: the consumer grads are identical
        and complete on every rank."""
        group, rank, _ = self._stream_tp
        N_l = hs_local.shape[2]
        h = g_pre[..., rank * N_l:(rank + 1) * N_l]                  # (b, s, N_l)
        partial = torch.einsum("bln,blnd->bld", h, hs_local)          # (b, s, h) partial
        return _TpAllReduce.apply(partial, group)                     # replicated

    def _sh_post(self, g_post, sublayer_partial):
        """Post broadcast on the rank's PARTIAL sublayer output (reviewer's
        fused scheme): the replicated post gates commute with the collective —
        Σ_p (g_post ⊙ o_p) = g_post ⊙ (Σ_p o_p) — so the Megatron
        o_proj/down all-reduce is ELIMINATED and the partial is summed
        inside the write-back reduce-scatter. Output: the FULL-N partial
        (b, s, N, h) (all N streams, this rank's sublayer contribution).
        Backward: the RS's all-gathered grad makes d(sublayer_partial) full
        (the standard rowwise backward contract) — no _TpSumGrad on the
        sublayer output. Requires Kr == 1 (temporal_aug off: TFA is
        nonlinear and does not commute with the partial sum)."""
        N, Kr = g_post.shape[2], g_post.shape[3]
        assert Kr == 1, "the fused partial-post scheme requires temporal_rank == 1"
        return torch.einsum("bln,bld->blnd", g_post.squeeze(-1), sublayer_partial)

    def _sh_res(self, g_res, residual, post_partial):
        """Fused write-back (reviewer's scheme): B_p = res_partial + post_partial
        (Σ_p B_p = H_res·X + g_post·Σ_p o_p — exact by linearity), then ONE
        reduce-scatter hands each rank its next-layer stream block."""
        group, rank, tp = self._stream_tp
        N_l = residual.shape[2]
        h_n = g_res[..., rank * N_l:(rank + 1) * N_l]                  # (b, s, N, N_l)
        partial = (torch.einsum("blmn,blnd->blmd", h_n, residual)
                   + post_partial)                                     # (b, s, N, h) partial
        return _TpReduceScatterStreams.apply(partial, group, rank, N_l)

    def _sh_res_legacy(self, g_res, residual):
        """LEGACY write-back's res term (identical einsum to _sh_res, without
        the post partial): (b, s, N, h) partial -> ONE reduce-scatter."""
        group, rank, tp = self._stream_tp
        N_l = residual.shape[2]
        h_n = g_res[..., rank * N_l:(rank + 1) * N_l]                  # (b, s, N, N_l)
        partial = torch.einsum("blmn,blnd->blmd", h_n, residual)       # (b, s, N, h)
        return _TpReduceScatterStreams.apply(partial, group, rank, N_l)

    def _sh_post_local(self, g_post, o):
        """LEGACY write-back: o is the SUMMED (replicated) sublayer output —
        the Megatron rowwise AR already happened — so post writes only this
        rank's m-block (b, s, N_l, h) with NO forward communication.

        _TpSumGrad on o is REQUIRED (the mhc-only autograd trap): o is
        replicated but consumed per-rank (each rank reads only its m-block),
        so the per-rank d(o) pieces are PARTIAL — the backward AR sums them
        before the rowwise o_proj/down backward. (The reviewer scheme makes
        this wrapper obsolete: its RS backward delivers the full grad.)"""
        from torch.distributed.tensor import DTensor
        group, rank, tp = self._stream_tp
        if isinstance(o, DTensor):
            o = o.to_local()                                           # RowwiseParallel Replicate output
        N, Kr = g_post.shape[2], g_post.shape[3]
        assert Kr == 1, "legacy write-back requires temporal_rank == 1"
        N_l = N // tp
        o = _TpSumGrad.apply(o, group)
        h = g_post[..., rank * N_l:(rank + 1) * N_l, 0]                # (b, s, N_l)
        return torch.einsum("blm,bld->blmd", h, o)                     # (b, s, N_l, h)

    def _forward_stream_shard(self, hidden_states, attention_mask, position_ids,
                              past_key_values, use_cache, cache_position,
                              position_embeddings, **kwargs):
        """Route-C forward: the residual state (b, s, N, h) is Shard(-2). The
        backbone stays REPLICATED on the collapsed single stream (Megatron
        colwise/rowwise with its o_proj/down all-reduces, unchanged); the
        mappings run partial with one gates all-reduce, one pre-read
        all-reduce and one res reduce-scatter per sublayer. Unlike tphc,
        mhc's stream sharding ADDS collectives (the collapsed sublayer input
        must be replicated) — it trades communication for memory/4 and
        mapping compute/4."""
        bsz, q_len, _, _ = hidden_states.size()

        residual = hidden_states
        x = residual.reshape(bsz, q_len, -1)   # raw flat; the norm is fused into _sh_norm_gates
        g_pre, g_post, g_res = self._sh_norm_gates(
            self.layernorm_attn, x, self.pre_mapping_attn, self.post_mapping_attn, self.res_mapping_attn)
        hs = self._sh_pre_read(g_pre, residual)
        hs = self.input_layernorm(hs)
        # Self Attention (replicated single stream; o_proj output stays PARTIAL
        # — its sum is folded into the write-back RS, no Megatron AR)
        attn_out, _ = self.self_attn(
            hidden_states=hs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if self._legacy_writeback:
            hidden_states = (self._sh_res_legacy(g_res, residual)
                             + self._sh_post_local(g_post, attn_out))
        else:
            hidden_states = self._sh_res(g_res, residual, self._sh_post(g_post, attn_out))

        # Fully Connected
        residual = hidden_states
        x = residual.reshape(bsz, q_len, -1)
        g_pre, g_post, g_res = self._sh_norm_gates(
            self.layernorm_mlp, x, self.pre_mapping_mlp, self.post_mapping_mlp, self.res_mapping_mlp)
        hs = self._sh_pre_read(g_pre, residual)
        hs = self.post_attention_layernorm(hs)
        hs = self.mlp(hs)
        if self._legacy_writeback:
            hidden_states = (self._sh_res_legacy(g_res, residual)
                             + self._sh_post_local(g_post, hs))
        else:
            hidden_states = self._sh_res(g_res, residual, self._sh_post(g_post, hs))
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        if self._stream_tp is not None:
            return self._forward_stream_shard(
                hidden_states, attention_mask, position_ids, past_key_values,
                use_cache, cache_position, position_embeddings, **kwargs)

        bsz, q_len, _, _ = hidden_states.size()

        residual = hidden_states
        x = self.layernorm_attn(hidden_states.reshape(bsz, q_len, -1))
        hidden_states = self.pre_mapping_attn(x, hidden_states)
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = self.post_mapping_attn(x, hidden_states)
        hidden_states = hidden_states + self.res_mapping_attn(x, residual)

        # Fully Connected
        residual = hidden_states
        x = self.layernorm_mlp(hidden_states.reshape(bsz, q_len, -1))
        hidden_states = self.pre_mapping_mlp(x, hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.temporal is not None:
            hidden_states = self.temporal(hidden_states)  # (b, s, Kr, C)
        hidden_states = self.post_mapping_mlp(x, hidden_states)
        hidden_states = hidden_states + self.res_mapping_mlp(x, residual)
        return hidden_states


class MyLlamaModel(LlamaModel):
    
    config_class = MyLlamaConfig

    def __init__(self, config: MyLlamaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [MyLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # Route-C stream sharding (set by apply_tensor_parallel): (group, rank, tp)
        self._stream_tp = None

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        if self._stream_tp is not None:
            group, rank, tp = self._stream_tp
            hidden_states = _TpExpandEntry.apply(
                inputs_embeds, group, self.config.num_streams // tp)
        else:
            hidden_states = hidden_states[:,:,None,:].expand(-1, -1, self.config.num_streams, -1).contiguous()

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        if self._stream_tp is not None:
            hidden_states = _TpCollapseExit.apply(hidden_states, self._stream_tp[0])
        else:
            hidden_states = hidden_states.sum(dim=2)
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class MyLlamaForCausalLM(LlamaForCausalLM):
    
    config_class = MyLlamaConfig

    def __init__(self, config: MyLlamaConfig):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = MyLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def apply_tensor_parallel(model, mesh, shard_streams: bool = False):
    """Tensor-parallelize an mhc-llama ``MyLlamaForCausalLM`` over a 1-D `mesh`.

    mhc's backbone sublayers are vanilla LlamaAttention/LlamaMLP applied to the
    collapsed single stream, which is replicated on every rank — so the standard
    Megatron plan applies unchanged:

      q/k/v, gate/up  colwise   (each rank holds whole heads / FFN blocks)
      o_proj, down    rowwise   (partial sums + one all-reduce per sublayer)
      lm_head         colwise with Replicate output (full-vocab logits for CE)

    Everything operating on the N-stream state stays replicated and needs no
    sharding: PreMapping collapses it, PostMapping broadcasts it, ResMapping
    (weight included — its Sinkhorn needs the full (N, N) mixing matrix),
    layernorm_attn/mlp are tiny, and the temporal convs sit on the (already
    all-reduced) MLP output.

    Call AFTER moving the model to the mesh device and with identical
    (same-seed) weights on every rank: distribute_tensor scatters each rank's
    local copy into its shard.
    """
    from torch.distributed.tensor import Replicate
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        parallelize_module,
    )

    tp_size = mesh.size()
    if tp_size == 1:
        # single-rank mesh: nothing to shard. Also avoids a torch-2.5 bug where
        # DTensor einsum/matmul strategy generation crashes on 1-rank meshes.
        return model
    cfg = model.config
    for name, value in (
        ("num_attention_heads", cfg.num_attention_heads),
        ("num_key_value_heads", cfg.num_key_value_heads),
    ):
        if value % tp_size != 0:
            raise ValueError(
                f"TP size {tp_size} must divide {name} ({value}); head boundaries "
                "must stay intact for the sharded attention reshape."
            )

    if shard_streams:
        # Route C, reviewer-fused variant: q/k/v/gate/up stay colwise, but
        # o_proj/down_proj are sharded WITHOUT the all-reduce
        # (PartialRowwiseLinear) — the sublayer output stays PARTIAL and the
        # post gates (replicated scalars) commute with the collective, so its
        # sum happens inside the write-back reduce-scatter instead of a
        # dedicated Megatron AR. Forward collectives per sublayer:
        # pre-read AR + write-back RS = 4.5C (was 6C with the o_proj AR).
        # Weight-shard grads: o_proj/down are PARAM-sharded (PartialRowwiseLinear
        # stores only the local column block — grad complete locally, no sync);
        # the merged-gate + norm slices are plain narrows synced via
        # post-accumulate-grad hooks (tier-1), no custom-Function graph breaks.
        if cfg.num_streams % tp_size != 0:
            raise ValueError(
                f"shard_streams requires num_streams ({cfg.num_streams}) divisible "
                f"by the TP size ({tp_size})."
            )
        if cfg.temporal_aug:
            raise ValueError(
                "shard_streams requires temporal_aug off: the TFA is nonlinear and "
                "does not commute with the partial sublayer output"
            )
        import torch.distributed as dist
        group = mesh.get_group()
        stream_tp = (group, dist.get_rank(group), tp_size)
        # MHC_LEGACY_WRITEBACK=1 selects the pre-reviewer write-back: o_proj /
        # down_proj keep the Megatron RowwiseParallel all-reduce (6C/sublayer)
        # and post writes only the local m-block — smaller write-back partials
        # than the reviewer scheme's full-(N·h) fused B_p (4.5C/sublayer).
        # Use it as the memory-optimized operating point.
        legacy = os.environ.get("MHC_LEGACY_WRITEBACK", "0") == "1"
        plan = {
            # NO o_proj / down_proj unless legacy — PartialRowwiseLinear below
            "model.layers.*.self_attn.q_proj": ColwiseParallel(),
            "model.layers.*.self_attn.k_proj": ColwiseParallel(),
            "model.layers.*.self_attn.v_proj": ColwiseParallel(),
            "model.layers.*.mlp.gate_proj": ColwiseParallel(),
            "model.layers.*.mlp.up_proj": ColwiseParallel(),
            "lm_head": ColwiseParallel(output_layouts=Replicate()),
        }
        if legacy:
            plan["model.layers.*.self_attn.o_proj"] = RowwiseParallel()
            plan["model.layers.*.mlp.down_proj"] = RowwiseParallel()
        model = parallelize_module(model, mesh, plan)
        model.model._stream_tp = stream_tp
        for layer in model.model.layers:
            layer._stream_tp = stream_tp
            layer._legacy_writeback = legacy
            if not legacy:
                layer.self_attn.o_proj = PartialRowwiseLinear(
                    layer.self_attn.o_proj, stream_tp, cfg.hidden_size // tp_size)
                layer.mlp.down_proj = PartialRowwiseLinear(
                    layer.mlp.down_proj, stream_tp, cfg.intermediate_size // tp_size)
            # tier-1: _sh_norm_gates slices the merged-gate and norm weights
            # with plain narrows — reassemble their block grads via hooks
            # (post-accumulate AR, outside the compiled regions).
            for sub in ("attn", "mlp"):
                _register_partial_grad_sync(
                    getattr(layer, f"layernorm_{sub}").weight, group)
                for kind in ("pre", "post", "res"):
                    _register_partial_grad_sync(
                        getattr(layer, f"{kind}_mapping_{sub}").weight, group)
        return model

    model = parallelize_module(
        model,
        mesh,
        {
            "model.layers.*.self_attn.q_proj": ColwiseParallel(),
            "model.layers.*.self_attn.k_proj": ColwiseParallel(),
            "model.layers.*.self_attn.v_proj": ColwiseParallel(),
            "model.layers.*.self_attn.o_proj": RowwiseParallel(),
            "model.layers.*.mlp.gate_proj": ColwiseParallel(),
            "model.layers.*.mlp.up_proj": ColwiseParallel(),
            "model.layers.*.mlp.down_proj": RowwiseParallel(),
            "lm_head": ColwiseParallel(output_layouts=Replicate()),
        },
    )
    return model
