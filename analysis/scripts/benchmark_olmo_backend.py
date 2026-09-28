"""One-off benchmark: does `allenai/OLMo-2-0425-1B` run fast enough for
realtime per-utterance embedding on this machine (Apple M2, 8GB RAM), and
does `mlx-lm` support the OLMo-2 architecture? Answers both empirically
instead of trusting docs/search results, which disagreed on hidden size and
mlx-lm coverage for this model.

This downloads the model and runs inference - per analysis/CLAUDE.md, run it
yourself rather than having Claude run it. Needs `mlx` + `mlx-lm` installed
in addition to text_embeddings.py's existing deps (torch, transformers):

    pip install mlx mlx-lm transformers

Usage:
    python analysis/scripts/benchmark_olmo_backend.py
"""

import sys
import time
from pathlib import Path

MODEL_NAME = "allenai/OLMo-2-0425-1B"

SAMPLE_TEXTS = [
    "It looks like a bird flying over the ocean at sunset.",
    "I think that's a person standing next to a tree.",
    "The colors remind me of autumn leaves.",
    "There's something strange about the shape in the middle.",
    "It could be a reflection in water.",
    "I'm not sure, maybe a mountain range in the background.",
    "That looks like two people talking to each other.",
    "The texture reminds me of sand or fabric.",
]


def bench_mlx():
    print("\n=== mlx-lm ===")
    try:
        import mlx.core as mx
        from mlx_lm import load
    except ImportError as e:
        print(f"mlx / mlx-lm not installed: {e}")
        return None

    t0 = time.time()
    try:
        model, tokenizer = load(MODEL_NAME)
    except Exception as e:
        print(f"mlx-lm could not load {MODEL_NAME}: {type(e).__name__}: {e}")
        return None
    print(f"load: {time.time() - t0:.1f}s")

    # mlx-lm causal-LM wrappers expose the bare transformer as `.model`
    # (embed_tokens -> layers -> norm), separate from `.lm_head`. That
    # output is the closest match to transformers' hidden_states[-1], which
    # text_embeddings.TextEmbedder reads by default (layer=-1).
    try:
        ids = mx.array([tokenizer.encode(SAMPLE_TEXTS[0])])
        t0 = time.time()
        hidden = model.model(ids)
        mx.eval(hidden)
        print(f"first forward pass: {time.time() - t0:.3f}s, hidden shape {hidden.shape}")
    except Exception as e:
        print("mlx-lm loaded the model but a forward pass through `.model` failed "
              f"(architecture likely unsupported yet): {type(e).__name__}: {e}")
        return None

    # One utterance at a time, no batching - matches realtime usage.
    times = []
    for t in SAMPLE_TEXTS:
        ids = mx.array([tokenizer.encode(t)])
        t0 = time.time()
        hidden = model.model(ids)
        mx.eval(hidden)
        times.append(time.time() - t0)
    print(f"per-utterance embed latency: mean {sum(times) / len(times) * 1000:.0f}ms, "
          f"max {max(times) * 1000:.0f}ms")
    print(f"hidden size: {hidden.shape[-1]}")
    return hidden.shape[-1]


def bench_transformers_mps():
    print("\n=== transformers (device=mps) ===")
    sys.path.insert(0, str(Path(__file__).parent))
    from text_embeddings import TextEmbedder

    t0 = time.time()
    embedder = TextEmbedder(model_name=MODEL_NAME, device="mps", dtype="float16")
    print(f"load: {time.time() - t0:.1f}s, hidden size {embedder.hidden_size}")

    embedder.embed([SAMPLE_TEXTS[0]], batch_size=1)  # warm up mps

    times = []
    for t in SAMPLE_TEXTS:
        t0 = time.time()
        embedder.embed([t], batch_size=1)
        times.append(time.time() - t0)
    print(f"per-utterance embed latency: mean {sum(times) / len(times) * 1000:.0f}ms, "
          f"max {max(times) * 1000:.0f}ms")
    embedder.unload()
    return embedder.hidden_size


if __name__ == "__main__":
    mlx_dim = bench_mlx()
    hf_dim = bench_transformers_mps()
    print("\n=== summary ===")
    print(f"mlx-lm hidden size: {mlx_dim}")
    print(f"transformers hidden size: {hf_dim}")
    if mlx_dim is not None and hf_dim is not None and mlx_dim != hf_dim:
        print("WARNING: hidden sizes disagree between backends - investigate "
              "before picking one, something's reading the wrong layer.")
