# Correctness test for the three cache strategies in evaluation/fast_dream_generate.py:
#   generate_embeds_dream                  (none)
#   generate_with_prefix_cache_embeds_dream (prefix)
#   generate_with_dual_cache_embeds_dream  (dual)
#
# KEY FACT: Dream is a BIDIRECTIONAL diffusion LM. With >=2 layers, a position's
# cached K/V depends (through prior layers' attention) on the still-masked answer
# tokens, so prefix/dual caching is an APPROXIMATION of the exact `none` decode —
# the cached modes can score slightly differently. That is a speed/accuracy
# trade-off, not a bug.
#
# What we CAN assert exactly: with a 1-LAYER model, each position's K/V is just a
# projection of that position's own embedding (no cross-position mixing), so
# caching is exact. If none/prefix/dual disagree on a 1-layer model there is a
# real plumbing bug (wrong RoPE slice, KV truncation, replace_position, or a
# missing EOS-fill re-warm). So:
#   * 1-layer  -> assert none == prefix == dual  (hard correctness check)
#   * 2-layer  -> report agreement rate (informational; divergence is expected)
#
# A small vocab + an EOS-logit bias make EOS appear during decoding, exercising
# the EOS-fill -> KV re-warm path in the cached modes.
#
# Run: python tests/test_cache_mode_equiv.py   (or: pytest tests/)
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["DLLM_VSR_USE_FAST_DREAM"] = "1"

import torch

from evaluation.fast_dream_model.configuration_dream import DreamConfig
from evaluation.fast_dream_model.modeling_dream import DreamModel
from evaluation.fast_dream_generate import (
    generate_embeds_dream,
    generate_with_prefix_cache_embeds_dream,
    generate_with_dual_cache_embeds_dream,
)

VOCAB, MASK_ID, EOS_ID, PAD_ID = 16, 15, 5, 6

GENS = {
    "none": generate_embeds_dream,
    "prefix": generate_with_prefix_cache_embeds_dream,
    "dual": generate_with_dual_cache_embeds_dream,
}

REGIMES = [
    dict(name="fixed_k, 2 blk", gen_length=16, steps=16, block_length=8,
         alg="fixed_k", threshold=None, fixed_k=1, factor=1.0),
    dict(name="fixed_k, 1 blk", gen_length=16, steps=16, block_length=16,
         alg="fixed_k", threshold=None, fixed_k=2, factor=1.0),
    dict(name="threshold, 2 blk", gen_length=16, steps=16, block_length=8,
         alg="threshold", threshold=0.5, fixed_k=1, factor=1.0),
    dict(name="dynamic, 4 blk", gen_length=16, steps=16, block_length=4,
         alg="dynamic", threshold=None, fixed_k=1, factor=0.5),
]


class StubTokenizer:
    eos_token_id = EOS_ID

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(i) for i in ids)


class EosBiasedModel:
    """Delegate to a real DreamModel but add a constant to the EOS logit so EOS is
    predicted during decoding (exercises the EOS-fill -> KV re-warm path). Every
    call/kwarg (dual_cache, replace_position, past_key_values, use_cache) is
    forwarded unchanged; only out.logits[..., EOS_ID] is bumped.
    """
    def __init__(self, model, bias=0.6):
        self._model, self._bias = model, bias

    def __getattr__(self, name):
        return getattr(self._model, name)

    def __call__(self, *args, **kwargs):
        out = self._model(*args, **kwargs)
        out.logits[..., EOS_ID] = out.logits[..., EOS_ID] + self._bias
        return out


def build_model(num_layers, seed=0):
    torch.manual_seed(seed)
    cfg = DreamConfig(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
        num_hidden_layers=num_layers, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256, rms_norm_eps=1e-6, rope_theta=10000.0,
        mask_token_id=MASK_ID, pad_token_id=PAD_ID, eos_token_id=EOS_ID,
    )
    return DreamModel(cfg).to(torch.float32).eval()


def decode_all(model, r):
    tok = StubTokenizer()
    instructions = [torch.tensor([1, 2, 3], dtype=torch.long)]
    post_user_ids = torch.tensor([4, 7], dtype=torch.long)
    out = {}
    for mode, fn in GENS.items():
        torch.manual_seed(7)
        v_feats = torch.randn(1, 4, 64)
        _, ids, nfe = fn(
            model=model, tokenizer=tok, instructions=instructions,
            v_feats=v_feats, post_user_prefix=post_user_ids, mask_id=MASK_ID,
            gen_length=r["gen_length"], steps=r["steps"], block_length=r["block_length"],
            alg=r["alg"], threshold=r["threshold"], fixed_k=r["fixed_k"], factor=r["factor"],
            tail_token_id=None,
        )
        out[mode] = (ids, nfe)
    return out


def run_one_layer_exact():
    """1-layer model must be bit-exact across cache modes. Returns True on pass."""
    all_ok = True
    for seed in range(4):
        plain = build_model(1, seed)
        eos = EosBiasedModel(build_model(1, seed))
        for tag, model in (("plain", plain), ("eos", eos)):
            for r in REGIMES:
                o = decode_all(model, r)
                ok = (o["none"][0] == o["prefix"][0] == o["dual"][0])
                all_ok = all_ok and ok
                if not ok:
                    print(f"FAIL seed={seed} {tag} {r['name']}")
                    for m in GENS:
                        print(f"    {m:>6}: {o[m][0]}")
    return all_ok


def test_one_layer_exact():
    assert run_one_layer_exact()


def main():
    # ---- hard check: 1-layer model must be bit-exact across cache modes ----
    print("=== 1-layer (caching is EXACT — hard assertion) ===")
    all_ok = run_one_layer_exact()
    print("1-layer exact equivalence:", "PASS" if all_ok else "FAIL")

    # ---- informational: 2-layer model is an approximation (no assertion) ----
    print("\n=== 2-layer (caching is APPROXIMATE — informational only) ===")
    agree_p = agree_d = tot = 0
    for seed in range(4):
        for tag, model in (("plain", build_model(2, seed)),
                           ("eos", EosBiasedModel(build_model(2, seed)))):
            for r in REGIMES:
                o = decode_all(model, r)
                tot += 1
                agree_p += (o["none"][0] == o["prefix"][0])
                agree_d += (o["none"][0] == o["dual"][0])
    print(f"agreement vs exact `none`:  prefix {agree_p}/{tot}   dual {agree_d}/{tot}")
    print("(divergence here is the expected bidirectional speed/accuracy trade-off)")

    print("\nRESULT:", "OK" if all_ok else "PLUMBING BUG")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
