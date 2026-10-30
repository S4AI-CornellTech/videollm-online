# Faster streaming inference for long KV caches:
#   PreallocatedCache   - one KV buffer per layer, allocated once; new tokens are written in place instead of
#                         torch.cat-ing the whole cache every frame (which copies it and fragments memory)
#   attn "sdpa_gqa"     - SDPA that uses the shared (grouped) KV heads directly, also when an attention mask is
#                         needed (per-frame prefill), instead of repeat_kv expanding K/V 4x
# Both compute the same attention as the defaults; only memory traffic and allocation change.
import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicLayer
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
from transformers.modeling_utils import AttentionInterface


class PreallocatedLayer(DynamicLayer):
    """KV for one layer in a buffer of `capacity` tokens, laid out (batch, capacity, heads, head_dim) so that the first
    `length` tokens are a dense prefix; keys / values are returned as (batch, heads, length, head_dim) views of it.
    Grows by doubling (one copy) only if `capacity` was too small."""
    capacity = 0  # set by PreallocatedCache

    def __init__(self, **kwargs):
        super().__init__()
        self.length = 0
        self.k_buf = self.v_buf = None

    def _views(self):
        self.keys = self.k_buf[:, :self.length].transpose(1, 2)
        self.values = self.v_buf[:, :self.length].transpose(1, 2)

    def update(self, key_states, value_states, cache_kwargs=None):
        batch, heads, n, dim = key_states.shape
        if self.k_buf is None:
            cap = max(self.capacity, n)
            self.k_buf = key_states.new_empty(batch, cap, heads, dim)
            self.v_buf = value_states.new_empty(batch, cap, heads, dim)
        if self.length + n > self.k_buf.shape[1]:
            cap = max(2 * self.k_buf.shape[1], self.length + n)
            k, v = self.k_buf.new_empty(batch, cap, heads, dim), self.v_buf.new_empty(batch, cap, heads, dim)
            k[:, :self.length] = self.k_buf[:, :self.length]
            v[:, :self.length] = self.v_buf[:, :self.length]
            self.k_buf, self.v_buf = k, v
        self.k_buf[:, self.length:self.length + n] = key_states.transpose(1, 2)
        self.v_buf[:, self.length:self.length + n] = value_states.transpose(1, 2)
        self.length += n
        self._views()
        return self.keys, self.values

    def get_seq_length(self, cache_position=None) -> int:
        return self.length

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = self.length - abs(max_length)
        if self.k_buf is not None and max_length < self.length:
            self.length = max_length
            self._views()


class PreallocatedCache(Cache):
    """Drop-in for DynamicCache with preallocated per-layer buffers. capacity: expected max tokens (frames x tokens per
    frame + text); exceeding it is allowed but costs one copy per doubling."""

    def __init__(self, capacity: int):
        layer = type('PreallocatedLayerCap', (PreallocatedLayer,), {'capacity': int(capacity)})
        super().__init__(layer_classes=layer)

    def buffer_bytes(self) -> int:
        return sum(l.k_buf.numel() * l.k_buf.element_size() * 2 for l in self.layers if getattr(l, 'k_buf', None) is not None)


def sdpa_gqa_attention_forward(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs):
    """Grouped-query SDPA without expanding K/V: the n_rep query heads that share a KV head are folded into the query
    length, (B, Hq, q, D) -> (B, Hkv, n_rep*q, D), and the mask is tiled to match. Exact same result as repeat_kv."""
    batch, q_heads, q_len, dim = query.shape
    kv_heads = key.shape[1]
    n_rep = q_heads // kv_heads
    if n_rep == 1 or (attention_mask is None and q_len > 1):
        # no grouping, or a causal prefill without a cache (first call): the default path is already fine
        return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)
    folded = query.reshape(batch, kv_heads, n_rep * q_len, dim)  # head h uses KV head h // n_rep, so groups are consecutive
    mask = attention_mask
    if mask is not None:
        mask = mask[:, :, :, :key.shape[-2]]
        mask = mask.repeat(1, 1, n_rep, 1) if mask.shape[1] == 1 else mask.reshape(batch, kv_heads, n_rep * q_len, -1)
    out = F.scaled_dot_product_attention(folded, key, value, attn_mask=mask,
                                         dropout_p=dropout if module.training else 0.0, scale=scaling)
    return out.reshape(batch, q_heads, q_len, dim).transpose(1, 2).contiguous(), None


# register under a new name; the mask must be registered too, or transformers silently runs without a mask
AttentionInterface.register('sdpa_gqa', sdpa_gqa_attention_forward)
AttentionMaskInterface.register('sdpa_gqa', sdpa_mask)
