# Check that PreallocatedCache + "sdpa_gqa" attention give the same outputs as the default DynamicCache + "sdpa" on the
# call pattern of streaming inference: one prefill, then 11-token frame prefills on top of the cache, then 1-token decodes.
# Usage (from the repo root, on a GPU node): python -m test.test_kv_cache
import torch
from transformers import LlamaConfig, LlamaForCausalLM, DynamicCache
from models.kv_cache import PreallocatedCache  # also registers "sdpa_gqa"

def run(model, cache, steps):
    outs = []
    for x in steps:
        o = model(inputs_embeds=x, past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        outs.append(o.logits.float())
    return outs, cache

def main():
    torch.manual_seed(0)
    cfg = LlamaConfig(hidden_size=256, intermediate_size=512, num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=2,
                      vocab_size=1000, max_position_embeddings=4096)
    base = LlamaForCausalLM(cfg).cuda().eval()
    hidden = cfg.hidden_size
    steps = [torch.randn(1, 37, hidden)] + [torch.randn(1, 11, hidden) for _ in range(40)] + [torch.randn(1, 1, hidden) for _ in range(10)]
    for dtype, tol in [(torch.float32, 1e-4), (torch.bfloat16, 5e-2)]:
        results = {}
        for name, attn, make_cache in [('default', 'sdpa', lambda: DynamicCache()),
                                       ('new', 'sdpa_gqa', lambda: PreallocatedCache(capacity=200))]:  # small capacity: also exercises growth
            model = base.to(dtype)
            model.config._attn_implementation = attn
            with torch.no_grad():
                results[name] = run(model, make_cache(), [s.cuda().to(dtype) for s in steps])
        (a, ca), (b, cb) = results['default'], results['new']
        diff = max((x - y).abs().max().item() for x, y in zip(a, b))
        print(f'{str(dtype):15s} max |logit diff| = {diff:.2e} (tolerance {tol:g}) -> {"OK" if diff < tol else "MISMATCH"}; '
              f'cache length default {ca.get_seq_length()} vs new {cb.get_seq_length()}')
        assert diff < tol and ca.get_seq_length() == cb.get_seq_length()
    print('all checks passed')

if __name__ == '__main__':
    main()
