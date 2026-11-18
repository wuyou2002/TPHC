from typing import List, Optional, Union, Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from transformers.utils.generic import check_model_inputs
from transformers.models.llama.modeling_llama import (
    eager_attention_forward,
    apply_rotary_pos_emb,
    LlamaMLP,
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaForCausalLM,
    LlamaRMSNorm
)
from .configuration_llama_tphc import MyLlamaConfig


logger = logging.get_logger(f"transformers.{__name__}")

def sinkhorn_knopps(log_alpha, iters = 20):
    dtype = log_alpha.dtype
    log_alpha = log_alpha.float()

    log_alpha = log_alpha - log_alpha.amax(dim = -2, keepdim = True).detach()

    alpha = log_alpha.exp()

    for _ in range(iters):
        alpha = F.normalize(alpha, p = 1, dim = -2)
        alpha = F.normalize(alpha, p = 1, dim = -1)

    return alpha.to(dtype)


class _TpSliceStreams(torch.autograd.Function):
    """Stream-dim (dim -2) TP scatter: keep only this rank's slice.

    Forward: local slice, zero communication. Backward: all-gather the
    per-rank gradient slices (rank order == stream order) so replicated
    upstream modules receive the FULL gradient — without this, layernorm /
    PreMapping / embedding grads would each hold only 1/tp of the streams
    and diverge across ranks.
    """

    @staticmethod
    def forward(ctx, x, group, rank, local_size):
        ctx.group, ctx.rank, ctx.local_size = group, rank, local_size
        return x[..., rank * local_size : (rank + 1) * local_size, :]

    @staticmethod
    def backward(ctx, grad):
        world = dist.get_world_size(ctx.group)
        buf = [torch.empty_like(grad) for _ in range(world)]
        dist.all_gather(buf, grad.contiguous(), group=ctx.group)
        return torch.cat(buf, dim=-2), None, None, None


class _TpGatherStreams(torch.autograd.Function):
    """Stream-dim (dim -2) TP gather: all-gather local slices into the full tensor.

    Forward: all-gather (rank order == stream order). Backward: keep only this
    rank's slice of the (replicated, hence identical) upstream gradient.
    """

    @staticmethod
    def forward(ctx, x, group, rank, local_size):
        ctx.group, ctx.rank, ctx.local_size = group, rank, local_size
        world = dist.get_world_size(group)
        buf = [torch.empty_like(x) for _ in range(world)]
        dist.all_gather(buf, x.contiguous(), group=group)
        return torch.cat(buf, dim=-2)

    @staticmethod
    def backward(ctx, grad):
        r, l = ctx.rank, ctx.local_size
        return grad[..., r * l : (r + 1) * l, :].contiguous(), None, None, None


class _TpSumGrad(torch.autograd.Function):
    """Identity forward; backward all-reduces the gradient.

    For replicated tensors consumed by a SHARDED computation: each rank
    computes only its local slice's contribution to the upstream gradient
    (e.g. PreMapping's sharded read: dL/dx and dL/dhs only sum this rank's
    fractions), so the true replicated-input gradient is the sum over ranks.
    """

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x.clone()   # fresh tensor, not the input as-is (compile/autograd view tracking)

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous().clone()
        dist.all_reduce(grad, group=ctx.group)
        return grad, None


class _TpAllReduce(torch.autograd.Function):
    """Forward all-reduce: y = sum_r x_r, shared by every rank. Backward is
    IDENTITY: y's grad flows unchanged to every rank's x_r (each contributed
    linearly; summing identical replicas would multiply by tp). Requires the
    CONSUMER-side partial grads to have been summed already (wrap the consumer
    with _TpSumGrad), or wrap THIS output with _TpSumGrad when each rank
    consumes the shared value with its own local data (then the arriving
    per-rank partials must be summed first). For TINY tensors (gate logits,
    norm statistics)."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        out = x.contiguous().clone()   # fresh tensor: all_reduce is in-place
        dist.all_reduce(out, group=group)
        return out

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class _TpKShardWeight(torch.autograd.Function):
    """A REPLICATED weight consumed row-block-wise (K-sharded): forward passes
    this rank's row block as a view (no communication); backward all-gathers
    every rank's block grad and concatenates — each rank's block IS the
    complete grad for those rows, so the concatenated full grad is identical
    on every rank (unlike _TpSumGrad, which SUMS and would mix row blocks)."""

    @staticmethod
    def forward(ctx, w, group, rank, rows):
        ctx.group = group
        lo, hi = rank * rows, (rank + 1) * rows
        block = w[lo:hi] if w.dim() == 1 else w[lo:hi, :]
        return block.clone()   # not a view of the param (compile/autograd view tracking)

    @staticmethod
    def backward(ctx, grad):
        world = dist.get_world_size(ctx.group)
        buf = [torch.empty_like(grad) for _ in range(world)]
        dist.all_gather(buf, grad.contiguous(), group=ctx.group)
        return torch.cat(buf, dim=0), None, None, None


class _TpReduceScatterStreams(torch.autograd.Function):
    """Stream-dim (dim -2) reduce-scatter: each rank holds a FULL-sized
    (b, s, K, h) PARTIAL tensor (its local-stream contribution for every K);
    rank r receives the fully reduced block r, (b, s, K_l, h).

    Forward: reduce_scatter (rank order == block order). Backward: every
    rank's partial receives the SAME full grad — the all-gather of the local
    output grads — so backward is a plain all-gather."""

    @staticmethod
    def forward(ctx, x, group, rank, local_size):
        # reduce_scatter_tensor / all_gather_into_tensor split along DIM 0 —
        # permute the stream dim (2) to the front and restore afterwards.
        # (The numel of the intended output coincidentally equals the dim-0
        # split's numel for common shapes, so getting this wrong fails SILENTLY.)
        ctx.group = group
        b, s, _, last = x.shape
        xT = x.permute(2, 0, 1, 3).contiguous()                     # (K, b, s, h)
        outT = torch.empty(local_size, b, s, last, device=x.device, dtype=x.dtype)
        dist.reduce_scatter_tensor(outT, xT, group=group)
        return outT.permute(1, 2, 0, 3).contiguous()                # (b, s, K_l, h)

    @staticmethod
    def backward(ctx, grad):
        b, s, K_l, last = grad.shape
        gT = grad.permute(2, 0, 1, 3).contiguous()                  # (K_l, b, s, h)
        world = dist.get_world_size(ctx.group)
        outT = torch.empty(K_l * world, b, s, last, device=grad.device, dtype=grad.dtype)
        dist.all_gather_into_tensor(outT, gT, group=ctx.group)
        return outT.permute(1, 2, 0, 3).contiguous(), None, None, None  # (b, s, K, h)


def _register_partial_grad_sync(param, group):
    """Grad reassembly for a REPLICATED param consumed BLOCK-wise or
    LOCALLY (only this rank's streams/fractions contribute) via plain tensor
    ops in the compiled graph — the tier-1 replacement for the _TpKShardWeight
    / _TpSumGrad(weight) custom Functions (one graph break per slice; the
    GEMMs and mapping einsums landed in separate compiled regions).

    Each rank's param grad is nonzero only in its block (narrow backward
    scatters into zeros, or the local-group consumer contributes only its
    part), so a post-accumulate-grad ALL-REDUCE reassembles the full grad on
    every rank — bitwise exact, because every element has at most one nonzero
    contribution per rank and disjoint rank blocks (x + 0 = x in IEEE). The
    hook runs in the autograd engine, OUTSIDE the compiled regions, where the
    old backward gather used to sit at a graph break.

    Fires once per backward: assumes NO gradient accumulation across
    multiple backwards per optimizer step (the hook would re-reduce the
    running total)."""
    if getattr(param, "_partial_grad_sync", None) is not None:
        return
    def _sync(p):
        if p.grad is not None:
            dist.all_reduce(p.grad, group=group)
    param._partial_grad_sync = param.register_post_accumulate_grad_hook(_sync)


class _TpExpandEntry(torch.autograd.Function):
    """Broadcast the replicated (b, s, h) embeddings into this rank's block of
    the N streams (all N blocks are equal at entry, so the local block IS the
    value — no communication). Backward: the true d_embeds is the sum over
    ALL N streams; each rank holds only its block's grad, so sum locally and
    all-reduce into the identical replicated d_embeds."""

    @staticmethod
    def forward(ctx, embeds, group, local_size):
        ctx.group = group
        return embeds.unsqueeze(2).expand(-1, -1, local_size, -1).contiguous()

    @staticmethod
    def backward(ctx, grad):
        g = grad.sum(dim=2).contiguous()
        dist.all_reduce(g, group=ctx.group)
        return g, None, None


class _TpCollapseExit(torch.autograd.Function):
    """Collapse the sharded stream state to the replicated (b, s, h): local
    partial sum over the rank's N_l streams, all-reduced. Backward: the
    (identical) replicated grad is broadcast back to the local block."""

    @staticmethod
    def forward(ctx, x_local, group):
        ctx.group, ctx.local = group, x_local.shape[2]
        out = x_local.sum(dim=2).contiguous()
        dist.all_reduce(out, group=group)
        return out

    @staticmethod
    def backward(ctx, grad):
        return grad.unsqueeze(2).expand(-1, -1, ctx.local, -1).contiguous(), None


class BatchedLinear(nn.Linear):
    r"""Similar to `torch.nn.Linear` but supports batched weights and biases.

    Examples::

        >>> m = BatchedLinear(5, 20, 30)
        >>> input = torch.randn(128, 5, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 5, 30])
    """

    __constants__ = ["batch_size", "in_features", "out_features"]
    batch_size: int

    def __init__(
        self,
        batch_size: int,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(in_features, out_features, bias, **factory_kwargs)
        self.batch_size = batch_size
        self.weight = nn.Parameter(
            torch.empty((batch_size, out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(batch_size, out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        # TP state (set by tp_shard); None = unsharded, plain forward.
        self._tp_mesh = None
        self._tp_slice_input = False
        self._tp_gather_output = False
        self._tp_stream_size = None

    def tp_shard(self, mesh, *, slice_input: bool = False, gather_output: bool = False,
                 stream_size: Optional[int] = None) -> None:
        """Shard this module's batch dim (the stream dim) across a 1-D TP mesh.

        weight/bias (B, ...) are distributed as Shard(0): each rank holds
        B/tp streams' full in->out projections. Because the streams are
        independent (no cross-stream contraction), the sharded bmm needs NO
        communication — the only collectives are the stream-dim
        scatter/gather at the module boundaries:

          slice_input   : the caller passes the FULL (..., B, in) tensor and
                          this module keeps only its local streams (backward
                          all-gathers the input gradient).
          gather_output : the module returns the FULL (..., B, out) tensor by
                          all-gathering the local streams (backward slices).

        `stream_size` overrides the sliced dim's size when the input's stream
        dim differs from `batch_size` (unused for the standard modules).
        """
        from torch.distributed.tensor import Shard, distribute_tensor

        if self.batch_size % mesh.size() != 0:
            raise ValueError(
                f"BatchedLinear batch_size ({self.batch_size}) must be divisible "
                f"by the TP mesh size ({mesh.size()})."
            )
        self._tp_mesh = mesh
        self._tp_slice_input = slice_input
        self._tp_gather_output = gather_output
        self._tp_stream_size = stream_size if stream_size is not None else self.batch_size
        self.weight = nn.Parameter(distribute_tensor(self.weight.data, mesh, [Shard(0)]))
        if self.bias is not None:
            self.bias = nn.Parameter(distribute_tensor(self.bias.data, mesh, [Shard(0)]))

    def _tp_forward(self, input: torch.Tensor) -> torch.Tensor:
        from torch.distributed.tensor import DTensor, Shard

        mesh = self._tp_mesh
        tp = mesh.size()
        group = mesh.get_group()
        rank = dist.get_rank(group)
        if self._tp_slice_input:
            local = self._tp_stream_size // tp
            input = _TpSliceStreams.apply(input, group, rank, local)
        *other, g_l, _ = input.shape  # g_l = local stream count after slicing
        x = input.reshape(-1, g_l, self.in_features).transpose(0, 1)   # (G_l, T, in)
        # weight is a Shard(0) DTensor: annotate x with the matching placement so
        # the bmm dispatches locally (Shard(0) x Shard(0) -> Shard(0), no comm).
        x = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)
        result = torch.bmm(x, self.weight.transpose(-1, -2))             # (G_l, T, out)
        if self.bias is not None:
            result = result + self.bias.unsqueeze(1)
        out = result.to_local().transpose(0, 1).reshape(*other, g_l, self.out_features)
        if self._tp_gather_output:
            out = _TpGatherStreams.apply(out, group, rank, g_l)
        return out

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Equivalent to ``einsum("...bi,boi->...bo", input, weight) + bias`` but uses
        `torch.bmm` with transposed views (no explicit permute/contiguous) for
        1) better performance, 2) tensor parallelism compatibility.
        """
        if self._tp_mesh is not None:
            return self._tp_forward(input)
        *other, _, _ = input.shape
        x = input.reshape(-1, self.batch_size, self.in_features).transpose(0, 1)  # (G, T, in)
        result = torch.bmm(x, self.weight.transpose(-1, -2))                       # (G, T, out)
        if self.bias is not None:
            result = result + self.bias.unsqueeze(1)                               # (G, 1, out)
        return result.transpose(0, 1).reshape(*other, self.batch_size, self.out_features)

    def extra_repr(self) -> str:
        return f"batch_size={self.batch_size}, in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"



class GroupedLlamaMLP(LlamaMLP):
    def __init__(self, config):
        super().__init__(config)
        self.intermediate_size = config.intermediate_size // config.num_groups
        # Fuse gate_proj + up_proj into a single batched projection (one bmm instead of two).
        self.gate_up_proj = BatchedLinear(config.num_groups, self.hidden_size, 2 * self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = BatchedLinear(config.num_groups, self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]
        del self.gate_proj, self.up_proj  # drop the full-size nn.Linear created by LlamaMLP.__init__

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)

    def tp_shard(self, mesh, slice_input: bool = True, gather_output: bool = True) -> None:
        """Stream-dim TP: gate_up slices its (full) input, down gathers its output.

        The SwiGLU elementwise op runs on the local streams; the only
        collectives are the scatter/gather at the projections.
        slice_input=False when the upstream PreMapping is TP-sharded and the
        input already contains only this rank's local streams.
        gather_output=False when the downstream PostMapping is stream-sharded
        (route C) and consumes only the local streams.
        """
        self.gate_up_proj.tp_shard(mesh, slice_input=slice_input)
        self.down_proj.tp_shard(mesh, gather_output=gather_output)


class GroupedLlamaAttention(LlamaAttention):
    """Grouped LLaMA attention module.

    Operates on num_groups streams. Supports two kv-head allocation modes:
    - Split: num_key_value_heads >= num_groups — each fraction gets
      num_key_value_heads // num_groups kv heads.
    - Tiled: num_groups > num_key_value_heads — only num_key_value_heads
      unique kv projection weights exist; they are tiled (repeated) across
      num_groups streams so that each fraction gets 1 kv head, with
      num_groups // num_key_value_heads fractions sharing each kv weight.
    """

    def __init__(self, config: MyLlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.num_groups = config.num_groups
        self.per_group_num_attention_heads = config.num_attention_heads // config.num_groups
        self.q_size = self.per_group_num_attention_heads * self.head_dim
        self.per_group_num_key_value_heads = config.num_key_value_heads // config.num_groups
        self.kv_size = self.per_group_num_key_value_heads * self.head_dim
        self.qkv_proj = BatchedLinear(
            config.num_groups, config.hidden_size,
            self.q_size + 2 * self.kv_size, bias=config.attention_bias,
        )
        self.o_proj = BatchedLinear(
            config.num_groups, self.q_size, config.hidden_size, bias=config.attention_bias,
        )
        del self.q_proj, self.k_proj, self.v_proj  # all merged into qkv_proj
        # TP state (set by tp_shard); None = unsharded.
        self._tp_mesh = None
        # True when the upstream PreMapping is TP-sharded and already feeds
        # this rank's local fractions -> skip the forward _TpSliceStreams.
        self._tp_pre_sliced = False

    def tp_shard(self, mesh, pre_sliced: bool = False, gather_output: bool = True) -> None:
        """Stream-dim TP for the grouped attention.

        Each rank takes num_groups/tp COMPLETE fractions (whole q/k/v head
        groups per fraction — no head-boundary constraints at all). The
        forward slices the (replicated) input to the local fractions; q/qkv
        projections consume the local input; o_proj all-gathers its output
        back to the full (b, s, G, h) so the mapping path stays replicated.

        pre_sliced=True: the upstream PreMapping is TP-sharded and already
        delivers this rank's local fractions -> skip the input slicing.
        """
        if self.num_groups % mesh.size() != 0:
            raise ValueError(
                f"num_groups ({self.num_groups}) must be divisible by the "
                f"TP mesh size ({mesh.size()})."
            )
        self._tp_pre_sliced = pre_sliced
        self._tp_mesh = mesh
        self.qkv_proj.tp_shard(mesh)               # input pre-sliced by forward
        self.o_proj.tp_shard(mesh, gather_output=gather_output)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._tp_mesh is not None and not self._tp_pre_sliced:
            # Stream-dim TP: keep only this rank's fractions (whole head groups;
            # backward all-gathers so the replicated upstream sees full grads).
            g_l = self.num_groups // self._tp_mesh.size()
            r = dist.get_rank(self._tp_mesh.get_group())
            hidden_states = _TpSliceStreams.apply(hidden_states, self._tp_mesh.get_group(), r, g_l)
        hidden_shape = (*hidden_states.shape[:-2], -1, self.head_dim)

        proj = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = proj.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )

        query_states = query_states.reshape(hidden_shape).transpose(1, 2)
        key_states = key_states.reshape(hidden_shape).transpose(1, 2)
        value_states = value_states.reshape(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

def get_update_groups(layer_idx: int, num_streams: int, num_updates: int) -> List[int]:
    """Compute which residual-stream groups are updated at layer `layer_idx`.

    Pattern (0-indexed, N=8, U=4):
        Layer 0: [0,1,2,3]  Layer 1: [0,1,4,5]  Layer 2: [0,1,6,7]  ...

    The first `fixed` groups are always included; the remaining `slide` groups
    cycle through the non-fixed range, advancing by `slide` each layer.
    """
    if num_streams <= num_updates:
        return list(range(num_streams))
    fixed = max(2 * num_updates - num_streams, num_updates // 2)
    fixed = min(fixed, num_updates)
    slide = num_updates - fixed
    available = num_streams - fixed
    if slide == 0 or available == 0:
        return list(range(num_updates))
    offset = (layer_idx * slide) % available
    sliding = [(offset + j) % available + fixed for j in range(slide)]
    return list(range(fixed)) + sliding


class PreMapping(nn.Module):
    def __init__(self, hidden_size: int, num_streams: int, num_groups: int, update_groups: List[int],
                 k: Optional[int] = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size * num_streams, num_groups * num_streams))
        bias = torch.full((num_groups, num_streams), -3.0)
        if len(update_groups) == num_groups:
            # standard sparse pattern (G == U): fraction i reads update group update_groups[i]
            for i, g in enumerate(update_groups):
                bias[i, g] = 3.0
        else:
            # dense regime (G < U = N): legacy dense init — fraction i reads
            # stream (k*G + i) % N, a sliding G-wide read window advancing by G
            # each layer (k = layer_idx % N).
            if k is None:
                raise ValueError(
                    "PreMapping legacy init (len(update_groups) != num_groups, i.e. G < U = N) "
                    "requires the layer window index k"
                )
            for i in range(num_groups):
                bias[i, (k * num_groups + i) % num_streams] = 3.0
        self.bias = nn.Parameter(bias)
        self.alpha = nn.Parameter(torch.ones(1) * 0.01)
        self.num_streams = num_streams
        self.num_groups = num_groups
        # 1-D TP mesh once tp_shard() has been called; None = unsharded.
        self._tp_mesh = None

    def tp_shard(self, mesh) -> None:
        """Shard the gate projection's OUTPUT dim (fractions) across the mesh.

        weight (N*h, G*N) -> Shard(1): rank r holds the gate columns of its
        own G/tp fractions (fraction f occupies columns [f*N, (f+1)*N), so
        Shard(1) blocks align with the stream-dim TP rank order). bias (G, N)
        -> Shard(0) on the fraction dim.

        Forward computes ONLY the local fractions' gated read -> (b, s, G_l, h):
        no communication in forward (x and hidden_states are replicated, and
        the local gate block reads all N streams locally). The downstream
        backbone (self_attn / mlp) then receives an already-local input and
        must skip its own _TpSliceStreams step (pre_sliced=True).

        Backward: dL/dx, dL/dhs and dL/dalpha are partial sums over the local
        fractions — _TpSumGrad all-reduces them. weight/bias grads are local
        to their shards (no comm needed).
        """
        from torch.distributed.tensor import Shard, distribute_tensor

        if self.num_groups % mesh.size() != 0:
            raise ValueError(
                f"num_groups ({self.num_groups}) must be divisible by the "
                f"TP mesh size ({mesh.size()})."
            )
        self._tp_mesh = mesh
        self.weight = nn.Parameter(distribute_tensor(self.weight.data, mesh, [Shard(1)]))
        self.bias = nn.Parameter(distribute_tensor(self.bias.data, mesh, [Shard(0)]))

    def forward(self, x: torch.Tensor, hidden_states: torch.Tensor):
        bsz, q_len, _ = x.size()
        if self._tp_mesh is not None:
            group = self._tp_mesh.get_group()
            g_l = self.num_groups // self._tp_mesh.size()
            # wrap the replicated inputs so their (partial) grads get summed
            x_in = _TpSumGrad.apply(x, group)
            hs_in = _TpSumGrad.apply(hidden_states, group)
            alpha = _TpSumGrad.apply(self.alpha, group)
            w = self.weight.to_local()   # (N*h, G_l*N); grad flows back into the Shard(1) DTensor param
            b = self.bias.to_local()     # (G_l, N)
            h = (x_in @ w).reshape(bsz, q_len, g_l, self.num_streams)
            h = F.sigmoid(alpha * h + b)
            return torch.einsum("blmn,blnd->blmd", h, hs_in)   # (b, s, G_l, h)
        h = self.alpha * torch.einsum("bld,dn->bln", x, self.weight).reshape(bsz, q_len, self.num_groups, self.num_streams) + self.bias
        h = F.sigmoid(h)
        return torch.einsum("blmn,blnd->blmd", h, hidden_states)

class PostMapping(nn.Module):
    def __init__(self, hidden_size: int, num_updates: int, update_groups: List[int],
                 num_groups: Optional[int] = None) -> None:
        super().__init__()
        if num_groups is None:
            num_groups = num_updates
        self.hidden_size = hidden_size
        self.num_updates = num_updates
        self.num_groups = num_groups
        # gate matrix (U, G): update stream m mixes the G sublayer output streams
        self.weight = nn.Parameter(torch.zeros(hidden_size * num_updates, num_updates * num_groups))
        if num_groups == num_updates:
            # standard sparse pattern (G == U): near-identity — update stream m
            # initially takes sublayer stream m only.
            bias = torch.full((num_updates, num_updates), -3.0) + 6.0 * torch.eye(num_updates)
        else:
            # dense regime (G < U = N): legacy dense init — fraction f writes
            # groups (f + j*G) % U for j in range(U // G), covering ALL update
            # groups each layer with exactly one open gate per group.
            bias = torch.full((num_updates, num_groups), -3.0)
            for f_idx in range(num_groups):
                for j in range(num_updates // num_groups):
                    bias[(f_idx + j * num_groups) % num_updates, f_idx] = 3.0
        self.bias = nn.Parameter(bias)
        self.alpha = nn.Parameter(torch.ones(1) * 0.01)
        self.register_buffer("update_groups", torch.tensor(update_groups))
        # Set True by the decoder layer when update_groups is the full stream
        # range (N == U dense regime): the index_select is an identity
        # permutation that still materializes a full copy — skip it.
        self.identity = False

    def forward(self, x: torch.Tensor, hidden_states: torch.Tensor):
        bsz, q_len, _ = x.size()
        # x: (b, s, N*h) → select U groups → (b, s, U*h)
        if self.identity:
            x_selected = x  # already (b, s, U*h) == (b, s, N*h)
        else:
            x_grouped = x.reshape(bsz, q_len, -1, self.hidden_size)  # (b, s, N, h)
            x_selected = x_grouped.index_select(2, self.update_groups).reshape(bsz, q_len, -1)  # (b, s, U*h)
        h = self.alpha * torch.einsum("bld,dn->bln", x_selected, self.weight).reshape(
            bsz, q_len, self.num_updates, self.num_groups) + self.bias
        h = F.sigmoid(h)
        # hidden_states: (b, s, G, h) sublayer output → (b, s, U, h) update streams
        return torch.einsum("blmn,blnd->blmd", h, hidden_states)


class ResMapping(nn.Module):
    def __init__(self, hidden_size: int, num_updates: int, update_groups: List[int]) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_updates = num_updates
        self.weight = nn.Parameter(torch.zeros(hidden_size * num_updates, num_updates * num_updates))
        self.bias = nn.Parameter(torch.full((num_updates, num_updates), -3.0) + 6.0 * torch.eye(num_updates))
        self.alpha = nn.Parameter(torch.ones(1) * 0.01)
        self.register_buffer("update_groups", torch.tensor(update_groups))
        # see PostMapping.identity: set by the decoder layer in the N == U
        # dense regime to skip the identity index_selects
        self.identity = False

    def forward(self, x: torch.Tensor, hidden_states: torch.Tensor):
        bsz, q_len, _ = x.size()
        # x: (b, s, N*h) → select U groups → (b, s, U*h)
        if self.identity:
            x_selected = x
            hs_selected = hidden_states
        else:
            x_grouped = x.reshape(bsz, q_len, -1, self.hidden_size)  # (b, s, N, h)
            x_selected = x_grouped.index_select(2, self.update_groups).reshape(bsz, q_len, -1)  # (b, s, U*h)
            # hidden_states: (b, s, N, h) → select U groups → (b, s, U, h)
            hs_selected = hidden_states.index_select(2, self.update_groups)  # (b, s, U, h)
        h = self.alpha * torch.einsum("bld,dn->bln", x_selected, self.weight).reshape(
            bsz, q_len, self.num_updates, self.num_updates) + self.bias
        h = sinkhorn_knopps(h)
        return torch.einsum("blmn,blnd->blmd", h, hs_selected)

def build_stream_shard_plan(update_groups: List[List[int]], num_streams: int, tp: int):
    """Where each residual stream lives under route C (stream-sharded TP), for
    every write (attention, MLP) of every layer. Pure Python, no communication.
    (Ported from modeling_llama_tphc2.py — the sparse-update extension of
    route C.)

    Two invariants make route C work for any N with sparse updates:
      * every rank always holds N/tp streams (balanced memory);
      * in every layer, every rank holds exactly U/tp of the U updated streams.
    The write is then ONE reduce-scatter of U streams for any N (the same
    volume as N == U): each rank receives U/tp new streams and drops its U/tp
    old ones.

    A fixed placement cannot keep the second invariant: the fixed streams are
    updated in every layer, so their ranks could never hold any other stream
    that is ever updated. Instead a stream may MOVE when it is written: the
    MLP write's reduce-scatter delivers each new stream to the rank that should
    hold it in the next layer. Streams only move inside a reduce-scatter that
    runs anyway, so moving costs no communication.

    Not a complete solver: greedy per-write placement with two fixed initial
    layouts (contiguous blocks, then round-robin); unsolvable schedules raise
    (fail-loud). Periodic schedules inherit solvability across layer counts.

    Returns plan[l] = [attn_step, mlp_step]; a step is a list over ranks of
    (layout, recv):
      layout : global stream id at each local position before the write
      recv   : update slots this rank receives from the write, in slot order
    After the write a rank holds [its non-updated streams, in layout order]
    + [update_groups[l][i] for i in recv].
    """
    L, N = len(update_groups), num_streams
    U = len(update_groups[0])
    if N % tp or U % tp:
        raise ValueError(f"route C needs num_streams ({N}) and num_updates ({U}) divisible by tp ({tp})")
    n_l, u_l = N // tp, U // tp

    def place(A, nxt, owner):
        """Receiving rank of every stream of A, so that the next layer's
        updated streams `nxt` end up U/tp per rank. Streams stay put when they can."""
        need = [u_l] * tp   # next layer's updated streams each rank still lacks
        cap = [u_l] * tp    # new streams each rank receives from this write
        inA, inN = set(A), set(nxt)
        for s in nxt:
            if s not in inA:            # not written now: keeps its rank
                need[owner[s]] -= 1
        if min(need) < 0:
            return None
        def fits(r, needed):
            return cap[r] > 0 and (need[r] > 0 or not needed)

        dest = {}
        # streams updated again in the next layer first, then the others
        for streams, needed in (([s for s in A if s in inN], True), ([s for s in A if s not in inN], False)):
            for s in streams:
                r = owner[s] if fits(owner[s], needed) else next(
                    (q for q in range(tp) if fits(q, needed)), None)
                if r is None:
                    return None
                dest[s] = r
                cap[r] -= 1
                if needed:
                    need[r] -= 1
        return dest

    def simulate(owner0):
        layout = [[s for s in range(N) if owner0[s] == r] for r in range(tp)]
        if any(len(lay) != n_l for lay in layout):
            return None
        plan = []
        for l in range(L):
            A = update_groups[l]
            slot = {s: i for i, s in enumerate(A)}
            steps = []
            for sub in range(2):
                owner = {s: r for r in range(tp) for s in layout[r]}
                if any(sum(owner[s] == r for s in A) != u_l for r in range(tp)):
                    return None
                if sub == 1 and l + 1 < L:
                    dest = place(A, update_groups[l + 1], owner)
                    if dest is None:
                        return None
                else:                    # attention write / last layer: keep ranks
                    dest = {s: owner[s] for s in A}
                recv = [sorted(slot[s] for s in A if dest[s] == r) for r in range(tp)]
                steps.append([(list(layout[r]), recv[r]) for r in range(tp)])
                layout = [[s for s in layout[r] if s not in slot] + [A[i] for i in recv[r]]
                          for r in range(tp)]
            plan.append(steps)
        return plan

    # contiguous blocks first (the original N == U layout), then round-robin
    for owner0 in ([s // n_l for s in range(N)], [s % tp for s in range(N)]):
        plan = simulate(owner0)
        if plan is not None:
            return plan
    raise ValueError(
        f"no balanced stream placement found for N={N}, U={U}, tp={tp} under this update schedule")


def stream_shard_step_indices(update_groups_l: List[int], step, rank: int) -> dict:
    """Index lists of one write of build_stream_shard_plan, for one rank."""
    U = len(update_groups_l)
    slot_of = {s: i for i, s in enumerate(update_groups_l)}
    layout = step[rank][0]
    slots = [slot_of.get(s, U) for s in layout]
    act_pos = [p for p, s in enumerate(layout) if s in slot_of]
    keep_pos = [p for p, s in enumerate(layout) if s not in slot_of]
    dest_order = [i for _, recv in step for i in recv]
    return {
        "streams": layout,                       # global stream id per local position
        "slots": slots,                          # update slot per local position (U: none)
        "act_pos": act_pos,                      # local positions of the updated streams
        "act_slots": [slots[p] for p in act_pos],
        "keep_pos": keep_pos,                    # local positions of the other streams
        "dest_order": dest_order,                # slot order of the write reduce-scatter
        "all_streams": [s for lay, _ in step for s in lay],
        "all_slots": [slot_of.get(s, U) for lay, _ in step for s in lay],
        "all_active": act_pos == list(range(len(layout))),
        "dest_identity": dest_order == list(range(U)),
    }


class _StreamShardStep:
    """Index tensors of one write under route C, for this rank. A plain object,
    not buffers: the indices differ per rank, so DDP buffer broadcasting must
    never see them. Built on the model's device by apply_tensor_parallel."""

    def __init__(self, idx: dict, device) -> None:
        def t(v):
            return torch.tensor(v, dtype=torch.long, device=device)
        self.streams, self.slots = t(idx["streams"]), t(idx["slots"])
        self.act_pos, self.act_slots = t(idx["act_pos"]), t(idx["act_slots"])
        self.keep_pos = t(idx["keep_pos"]) if idx["keep_pos"] else None
        self.dest_order = t(idx["dest_order"])
        self.all_streams, self.all_slots = t(idx["all_streams"]), t(idx["all_slots"])
        self.all_active, self.dest_identity = idx["all_active"], idx["dest_identity"]


class GroupedLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: MyLlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.config = config

        self.self_attn = GroupedLlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = GroupedLlamaMLP(config)
        update_groups = get_update_groups(layer_idx, config.num_streams, config.num_updates)
        self.update_groups = update_groups
        # Functional scatter: (N, U) one-hot matrix + (N,) mask, all pure-functional ops.
        # Avoids in-place index_copy_/scatter_ which break aot_autograd backward compilation.
        N, U = config.num_streams, config.num_updates
        scatter_mat = torch.zeros(N, U)
        for i, g in enumerate(update_groups):
            scatter_mat[g, i] = 1.0
        self.register_buffer("_scatter_mat", scatter_mat)          # (N, U)
        self.register_buffer("_update_mask", scatter_mat.sum(dim=1))  # (N,)
        # Dense N == U regime: update_groups is the full range, so the
        # mapping index_selects, the one-hot scatter einsum and the update
        # mask are all identity operations that still copy — flag them so
        # forward uses the original tensors directly (buffers stay
        # registered for state_dict compatibility).
        identity_update = update_groups == list(range(N))
        self._identity_update = identity_update
        # legacy read-window position (used only in the dense G < U = N regime)
        k = layer_idx % config.num_streams
        self.pre_mapping_attn = PreMapping(config.hidden_size, config.num_streams, config.num_groups, update_groups, k=k)
        self.post_mapping_attn = PostMapping(config.hidden_size, config.num_updates, update_groups, num_groups=config.num_groups)
        self.res_mapping_attn = ResMapping(config.hidden_size, config.num_updates, update_groups)
        self.post_mapping_attn.identity = identity_update
        self.res_mapping_attn.identity = identity_update
        self.pre_mapping_mlp = PreMapping(config.hidden_size, config.num_streams, config.num_groups, update_groups, k=k)
        self.post_mapping_mlp = PostMapping(config.hidden_size, config.num_updates, update_groups, num_groups=config.num_groups)
        self.res_mapping_mlp = ResMapping(config.hidden_size, config.num_updates, update_groups)
        self.post_mapping_mlp.identity = identity_update
        self.res_mapping_mlp.identity = identity_update
        self.layernorm_attn = LlamaRMSNorm(config.hidden_size * config.num_streams, eps=config.rms_norm_eps)
        self.layernorm_mlp = LlamaRMSNorm(config.hidden_size * config.num_streams, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx
        # Route-C stream sharding (set by apply_tensor_parallel): (group, rank, tp)
        self._stream_tp = None
        # Route-C placement of this layer's two writes (sparse updates, N > U):
        # (attn, mlp) _StreamShardStep. None in the dense N == U regime, whose
        # fixed contiguous layout needs no indices.
        self._sh_steps = None
        # self._init_weights()

    # ---------------- route C: stream-sharded forward (semantics preserved) ----------------

    def _sh_norm_gates(self, norm, x_local, pre, post, res, step=None):
        """Wide-norm statistics + all three mapping gate projections in ONE
        all-reduce (was: a separate (b, s) stats AR + a logits AR — one fewer
        collective and dynamo graph break per sublayer).

        Linearity makes the fusion exact: the gates consume
        einsum((x * inv * w), W) == inv * einsum((x * w), W), so the partial
        logits are computed from the UN-normalized local block with the norm
        weight folded in, and the norm scale `inv` is applied after the single
        all-reduce that also carries the sum-of-squares. The norm OUTPUT is
        never materialized (nothing else consumes it in the stream path).

        step is None in the dense N == U regime (contiguous layout — the tier-1
        plain-narrow path). With sparse updates (N > U) the local streams
        follow build_stream_shard_plan: Pre reads all N streams (rows gathered
        by GLOBAL STREAM id), Post/Res only the U updated ones (rows gathered
        by update SLOT; non-updated local positions read a ZERO row — the
        logit contribution they would make is 0, so the post-AR sum is exact).
        Still tier-1: plain index_select / F.pad ops (dynamo-traceable, no
        custom Function); each real stream/slot row is read by exactly ONE
        rank, so the post-accumulate AR hooks reassemble the grads bitwise."""
        bsz, q_len, _ = x_local.shape
        group, rank, tp = self._stream_tp
        N, G, U = self.config.num_streams, self.config.num_groups, self.config.num_updates
        K_l = x_local.shape[-1]
        cols = G * N + U * G + U * U
        if step is None:
            # tier-1 (ported from mhc): PLAIN narrows instead of
            # _TpKShardWeight — dynamo-traceable, no custom-Function graph
            # break; the block grads reassemble in post-accumulate-grad
            # all-reduce hooks registered at shard setup.
            lo, hi = rank * K_l, (rank + 1) * K_l
            w = norm.weight[lo:hi]                                         # (K_l,)
            W = torch.cat([pre.weight, post.weight, res.weight], dim=1)[lo:hi, :]
        else:
            h = self.config.hidden_size
            w = (norm.weight.view(N, h)
                 .index_select(0, step.streams).reshape(-1))              # (K_l,)
            W_pre = (pre.weight.view(N, h, G * N)
                     .index_select(0, step.streams)
                     .reshape(K_l, G * N))                                # rows: stream id
            # zero row U (appended via F.pad) for local positions that are NOT
            # updated in this layer — post/res gates only see the U updated
            # streams. pad backward = slice, so the zero-row grad vanishes.
            def slot_rows(weight, c):
                padded = F.pad(weight.view(U, h, c), (0, 0, 0, 0, 0, 1))   # (U+1, h, c)
                return padded.index_select(0, step.slots).reshape(K_l, c)
            W_post = slot_rows(post.weight, U * G)                        # rows: update slot
            W_res = slot_rows(res.weight, U * U)
            W = torch.cat([W_pre, W_post, W_res], dim=1)                   # (K_l, cols)
        # Fold the norm weight INTO the matmul weight: einsum((x*w), W) ==
        # einsum(x, W*w[:,None]) — the full-width (b,s,K_l) activation multiply
        # becomes a (K_l, cols) weight multiply (40x fewer elements, and its
        # backward too). Grad paths: d(W_block) = d(Wf)*w, d(w_block) =
        # (d(Wf)*W).sum(1) — partial (block rows only), completed by the hooks.
        Wf = W * w.unsqueeze(-1)
        partial_raw = torch.einsum("bld,dn->bln", x_local, Wf).float()      # (b, s, cols)
        # sumsq: bf16 squares + fp32 accumulation (sum(dtype=)) — avoids the
        # (b,s,K_l) fp32 materialization; per-element square rounding averages
        # out over the K_l-sum (~0.006% for K_l=4096) — negligible vs bf16.
        sumsq = x_local.square().sum(dim=-1, dtype=torch.float32)          # (b, s)
        # ONE all-reduce carries both (fp32: the partial logits are upcast —
        # exact, no precision loss)
        merged = _TpAllReduce.apply(
            torch.cat([partial_raw, sumsq.unsqueeze(-1)], dim=-1), group)
        raw_u, ss = merged.split([cols, 1], dim=-1)
        # each rank consumes the shared stats with its LOCAL slice -> the
        # partial grads must be summed before flowing back (the raw part
        # needs no wrap: its consumers pre-sum via the gates _TpSumGrad)
        ss = _TpSumGrad.apply(ss, group)
        inv = torch.rsqrt(ss.squeeze(-1) / norm.weight.shape[0] + norm.variance_epsilon)
        raw = (raw_u * inv.unsqueeze(-1)).to(x_local.dtype)
        raw_pre, raw_post, raw_res = raw.split([G * N, U * G, U * U], dim=-1)
        g_pre = torch.sigmoid(pre.alpha * raw_pre.view(bsz, q_len, G, N) + pre.bias)
        g_post = torch.sigmoid(post.alpha * raw_post.view(bsz, q_len, U, G) + post.bias)
        g_res = sinkhorn_knopps(res.alpha * raw_res.view(bsz, q_len, U, U) + res.bias)
        gates = _TpSumGrad.apply(torch.cat([
            g_pre.reshape(bsz, q_len, -1), g_post.reshape(bsz, q_len, -1),
            g_res.reshape(bsz, q_len, -1)], dim=-1), group)
        gp, go, gr = gates.split([G * N, U * G, U * U], dim=-1)
        return (gp.view(bsz, q_len, G, N), go.view(bsz, q_len, U, G), gr.view(bsz, q_len, U, U))

    def _sh_streamnorm(self, norm, x):
        """Per-stream RMSNorm over the last dim h with a sharded (b, s, G_l, h)
        input: statistics are local; the replicated weight is used DIRECTLY
        (tier-1: no _TpSumGrad graph break) and its grad — partial, only the
        local groups contribute on each rank — completes via a
        post-accumulate-grad all-reduce hook registered at shard setup."""
        xf = x.float()
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        out = (xf * torch.rsqrt(var + norm.variance_epsilon)).to(x.dtype)
        return out * norm.weight

    def _sh_pre(self, g_pre, hs_local, step=None):
        """Sharded PreMapping read: partials for ALL G fractions from the local
        N_l streams, ONE reduce-scatter hands each rank its (b, s, G_l, h)
        block. Gates come precomputed from _sh_norm_gates. step=None: dense
        contiguous layout (narrow); sparse: gather the gate columns of THIS
        rank's streams (g_pre is the complete post-AR (b, s, G, N))."""
        group, rank, tp = self._stream_tp
        G = self.config.num_groups
        if step is None:
            N_l = hs_local.shape[2]
            h_gate = g_pre[..., rank * N_l:(rank + 1) * N_l]
        else:
            h_gate = g_pre.index_select(-1, step.streams)            # (b, s, G, N_l)
        partial = torch.einsum("blgn,blnd->blgd", h_gate, hs_local)  # (b, s, G, h) partials
        return _TpReduceScatterStreams.apply(partial, group, rank, G // tp)

    def _sh_post_res(self, g_post, g_res, sublayer_out, residual, step=None):
        """Sharded Post+Res write-back with precomputed gates (see
        _sh_norm_gates): per-rank partials (post over the local fractions, res
        over the local updated streams), summed before ONE reduce-scatter.

        step=None (dense N == U): every local stream is updated, output rows
        are rank-major identity. Sparse (N > U): only the act_pos streams are
        res-mixed and written; when the plan moves streams (dest_order not
        identity) the RS output rows are permuted so each rank receives its
        destined NEW streams — the migration rides inside the collective. The
        non-updated (keep) streams are carried over with a plain index_select
        and cat, restoring the rank's N_l streams."""
        group, rank, tp = self._stream_tp
        G, U = self.config.num_groups, self.config.num_updates
        G_l = sublayer_out.shape[2]
        g_p = g_post[..., rank * G_l:(rank + 1) * G_l]
        if step is None:
            N_l = residual.shape[2]
            g_r = g_res[..., rank * N_l:(rank + 1) * N_l]
            partial = (torch.einsum("blun,blnd->blud", g_r, residual)
                       + torch.einsum("blug,blgd->blud", g_p, sublayer_out))
            return _TpReduceScatterStreams.apply(partial, group, rank, U // tp)
        g_r = g_res.index_select(-1, step.act_slots)                 # (b, s, U, U_l)
        if not step.dest_identity:
            g_p = g_p.index_select(2, step.dest_order)
            g_r = g_r.index_select(2, step.dest_order)
        res_act = residual if step.all_active else residual.index_select(2, step.act_pos)
        partial = (torch.einsum("blun,blnd->blud", g_r, res_act)
                   + torch.einsum("blug,blgd->blud", g_p, sublayer_out))
        new = _TpReduceScatterStreams.apply(partial, group, rank, U // tp)
        if step.keep_pos is None:                                    # N == U: all updated
            return new
        return torch.cat([residual.index_select(2, step.keep_pos), new], dim=2)

    def _forward_stream_shard(self, hidden_states, attention_mask, position_ids,
                              past_key_values, use_cache, cache_position,
                              position_embeddings, **kwargs):
        """Route-C forward: the residual state (b, s, N, h) is Shard(-2) across
        the TP mesh — each rank owns its block of streams end-to-end. Mapping
        gate projections run partial (K-sharded) with tiny logits all-reduces;
        the read/write mixes run as per-rank partials + reduce-scatter; the
        backbone consumes only the local fractions (o_proj/down no longer
        all-gather). Any N: with sparse updates (N > U) the local streams
        follow build_stream_shard_plan (N/tp per rank, U/tp of them updated
        per layer; the MLP write moves streams between ranks inside its RS).
        Semantics equal to the replicated path up to reduction order."""
        bsz, q_len, _, _ = hidden_states.size()
        # None in the dense regime — the fixed contiguous layout needs no
        # indices; the dense code path (step=None) is bit-identical to the
        # pre-sparse implementation.
        st_attn, st_mlp = self._sh_steps if self._sh_steps is not None else (None, None)

        residual = hidden_states
        # raw (un-normalized) flat residual: the wide norm is fused into the
        # gate projection (_sh_norm_gates) — its output is never materialized
        x = residual.reshape(bsz, q_len, -1)
        g_pre, g_post, g_res = self._sh_norm_gates(
            self.layernorm_attn, x, self.pre_mapping_attn, self.post_mapping_attn,
            self.res_mapping_attn, st_attn)
        hs = self._sh_pre(g_pre, residual, st_attn)
        hs = self._sh_streamnorm(self.input_layernorm, hs)
        # Self Attention (local fractions in, local out — no gather)
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
        hidden_states = self._sh_post_res(g_post, g_res, attn_out, residual, st_attn)

        # Fully Connected
        residual = hidden_states
        x = residual.reshape(bsz, q_len, -1)
        g_pre, g_post, g_res = self._sh_norm_gates(
            self.layernorm_mlp, x, self.pre_mapping_mlp, self.post_mapping_mlp,
            self.res_mapping_mlp, st_mlp)
        hs = self._sh_pre(g_pre, residual, st_mlp)
        hs = self._sh_streamnorm(self.post_attention_layernorm, hs)
        hs = self.mlp(hs)
        # the MLP write also moves streams to their next-layer ranks
        hidden_states = self._sh_post_res(g_post, g_res, hs, residual, st_mlp)
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
            # route C: stream-sharded residual state — see _forward_stream_shard
            return self._forward_stream_shard(
                hidden_states, attention_mask, position_ids, past_key_values,
                use_cache, cache_position, position_embeddings, **kwargs)
        # residual = hidden_states
        # hidden_states = self.input_attn_fusion(hidden_states)
        # hidden_states = self.input_layernorm(hidden_states)
        bsz, q_len, _, _ = hidden_states.size()
        N = self.config.num_streams
        h = self.config.hidden_size

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
        # PostMapping + ResMapping: only update selected groups (pure functional, no in-place ops)
        update = self.res_mapping_attn(x, residual) + self.post_mapping_attn(x, hidden_states)  # (b, s, U, h)
        if self._identity_update:
            # N == U: every group is updated — the one-hot scatter einsum and
            # the update mask are identities; write back directly.
            hidden_states = update
        else:
            update_full = torch.einsum("gu,bsuh->bsgh", self._scatter_mat, update)                # (b, s, N, h)
            mask = self._update_mask.view(1, 1, -1, 1)                                              # (1, 1, N, 1)
            hidden_states = residual * (1 - mask) + update_full

        # Fully Connected
        residual = hidden_states
        x = self.layernorm_mlp(hidden_states.reshape(bsz, q_len, -1))
        hidden_states = self.pre_mapping_mlp(x, hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        update = self.res_mapping_mlp(x, residual) + self.post_mapping_mlp(x, hidden_states)  # (b, s, U, h)
        if self._identity_update:
            hidden_states = update
        else:
            update_full = torch.einsum("gu,bsuh->bsgh", self._scatter_mat, update)                # (b, s, N, h)
            hidden_states = residual * (1 - mask) + update_full
        return hidden_states


class MyLlamaModel(LlamaModel):

    config_class = MyLlamaConfig

    def __init__(self, config: MyLlamaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [GroupedLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # Route-C stream sharding (set by apply_tensor_parallel): (group, rank, tp)
        self._stream_tp = None
        # `fusion` is unused by forward (all call sites commented out) but MUST
        # stay in the module tree exactly as before: old checkpoints' optimizer
        # states were saved with these params at these indices. DDP tolerance of
        # the unused params is handled by --ddp_find_unused_parameters true.
        # self.fusion = nn.Linear(config.num_streams, 1)

        # Initialize weights and apply final processing
        self.post_init()

    def post_init(self):
        """
        Post initialization processing.
        """
        super().post_init()
        if hasattr(self, "fusion"):
            self.fusion.weight.data.fill_(getattr(self.config, "input_fusion_weight", 1.0))

    def _init_weights(self, module):
        """
        Initialize the weights.
        """
        super()._init_weights(module)

        if hasattr(self.config, "initializer_range"):
            std = self.config.initializer_range
        else:
            # 0.02 is the standard default value across the library
            std = getattr(self.config.get_text_config(), "initializer_range", 0.02)

        if isinstance(module, BatchedLinear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()

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
        # hidden_states = self.fusion(hidden_states.mT).squeeze(-1).contiguous()
        hidden_states = self.norm(hidden_states)
        # hidden_states = self.fusion(hidden_states.mT).squeeze(-1).contiguous()

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


def apply_tensor_parallel(model, mesh, shard_pre_mapping: bool = False, shard_streams: bool = False):
    """Tensor-parallelize an tphc-llama model over a 1-D `mesh`.

    This variant's backbone runs G independent streams through BatchedLinear
    (bmm), so the natural TP is a STREAM-dim split — the "batched" strategy the
    bmm implementation was designed for:

      * every rank holds G/tp COMPLETE fractions (whole q/k/v head groups per
        fraction — no head-divisibility constraint, unlike Megatron head split)
      * q/kv/gate_up projections consume the local fractions (no comm)
      * attention + SwiGLU run entirely on the local fractions (no comm)
      * o_proj / down_proj all-gather their output back to the full (b, s, G, h)
      * _TpSliceStreams' backward all-gathers the input grads so replicated
        upstream modules (layernorm, mappings, embeddings) get full gradients

    Collectives per layer: 2 all-gathers (attn out, MLP out), each b*s*G*h
    elements — the same volume as the 2 all-reduces Megatron head-split TP
    would need (an all-gather is ~half the cost of an all-reduce).

    Everything else stays replicated and untouched: the N-stream residual
    state, PreMapping (N*h, G*N), Post/ResMapping (U*h, U*U — small, and
    Sinkhorn/index_select need complete matrices), the scatter buffers,
    embeddings and norms. lm_head is colwise-sharded with replicated output so the CE
    loss sees the full vocab.

    shard_pre_mapping=True additionally shards the PreMapping gate projection
    along its output (fraction) dim, so each rank computes only its local
    fractions' gated read — the backbone then gets an already-local input and
    skips the forward _TpSliceStreams. Backward all-reduces the partial
    x / hidden_states / alpha gradients (see PreMapping.tp_shard).

    Constraints: num_groups % tp == 0 and vocab_size % tp == 0.
    Call AFTER moving the model to the mesh device and with identical
    (same-seed) weights on every rank.

    NOTE: designed for training (use_cache=False); the KV cache under TP would
    store per-rank local fractions and needs extra handling for generation.
    """
    from torch.distributed.tensor import Replicate
    from torch.distributed.tensor.parallel import ColwiseParallel, parallelize_module

    tp_size = mesh.size()
    if tp_size == 1:
        # single-rank mesh: nothing to shard (also avoids 1-rank DTensor bugs).
        return model
    cfg = model.config
    if cfg.num_groups % tp_size != 0:
        raise ValueError(
            f"TP size {tp_size} must divide num_groups ({cfg.num_groups}): "
            "the stream-dim split keeps whole fractions per rank."
        )
    if cfg.vocab_size % tp_size != 0:
        raise ValueError(
            f"TP size {tp_size} must divide vocab_size ({cfg.vocab_size}) for the "
            "colwise lm_head shard."
        )

    if shard_streams:
        # Route C: shard the residual stream dim itself. Requires N, U and G
        # divisible by tp; works for the dense N == U regime AND for sparse
        # updates (N > U). The mapping gate projections run partial (tiny
        # logits all-reduce), the read/write mixes become reduce-scatters, and
        # o_proj/down stop all-gathering (the Post/Res partials consume only
        # local fractions). With sparse updates the streams follow
        # build_stream_shard_plan: each rank holds N/tp streams, U/tp of them
        # updated in every layer, and the MLP write moves streams between
        # ranks inside its reduce-scatter — so the collectives match N == U
        # (the write is ONE reduce-scatter of U streams for ANY N). Not
        # compatible with shard_pre_mapping (Pre is handled by the stream
        # path).
        N, G, U = cfg.num_streams, cfg.num_groups, cfg.num_updates
        if N % tp_size != 0 or G % tp_size != 0 or U % tp_size != 0:
            raise ValueError(
                f"shard_streams requires num_streams ({N}), num_groups ({G}) and "
                f"num_updates ({U}) divisible by the TP size ({tp_size})."
            )
        if shard_pre_mapping:
            raise ValueError("shard_streams and shard_pre_mapping are mutually exclusive")
        group = mesh.get_group()
        rank = dist.get_rank(group)
        stream_tp = (group, rank, tp_size)
        model.model._stream_tp = stream_tp
        layers = list(model.model.layers)
        if any(len(layer.update_groups) != U for layer in layers):
            raise ValueError(f"shard_streams requires num_updates ({U}) updated streams per layer")
        if U != N:
            # sparse updates: solve the stream placement (moves streams inside
            # the MLP write's reduce-scatter; fails loud if unsolvable)
            plan = build_stream_shard_plan([layer.update_groups for layer in layers], N, tp_size)
            for layer, steps in zip(layers, plan):
                device = layer.layernorm_attn.weight.device
                layer._sh_steps = tuple(
                    _StreamShardStep(stream_shard_step_indices(layer.update_groups, st, rank), device)
                    for st in steps)
        for layer in layers:
            if U == N and not layer._identity_update:
                raise ValueError("shard_streams requires the identity update regime")
            layer._stream_tp = stream_tp
            layer.self_attn.tp_shard(mesh, pre_sliced=True, gather_output=False)
            layer.mlp.tp_shard(mesh, slice_input=False, gather_output=False)
            # tier-1 (ported from mhc): the weight slices in _sh_norm_gates are
            # plain narrows and the per-stream norm weight is used directly —
            # reassemble their partial grads via post-accumulate hooks (outside
            # the compiled regions; -3 graph breaks per sublayer).
            for sub in ("attn", "mlp"):
                _register_partial_grad_sync(
                    getattr(layer, f"layernorm_{sub}").weight, group)
                for kind in ("pre", "post", "res"):
                    _register_partial_grad_sync(
                        getattr(layer, f"{kind}_mapping_{sub}").weight, group)
            _register_partial_grad_sync(layer.input_layernorm.weight, group)
            _register_partial_grad_sync(layer.post_attention_layernorm.weight, group)
        return model

    model = parallelize_module(
        model,
        mesh,
        {"lm_head": ColwiseParallel(output_layouts=Replicate())},
    )
    for layer in model.model.layers:
        if shard_pre_mapping:
            # Shard the PreMapping gate projection along its output (fraction)
            # dim: each rank computes only its local fractions' gated read, and
            # the backbone receives an already-local input (pre_sliced /
            # slice_input=False). Removes the replicated full-width pre_mapping
            # compute and the (b, s, G, h) activations it materialized per rank.
            layer.pre_mapping_attn.tp_shard(mesh)
            layer.pre_mapping_mlp.tp_shard(mesh)
        layer.self_attn.tp_shard(mesh, pre_sliced=shard_pre_mapping)
        layer.mlp.tp_shard(mesh, slice_input=not shard_pre_mapping)
    return model
