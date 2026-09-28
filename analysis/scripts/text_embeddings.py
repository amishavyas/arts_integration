"""OLMo embeddings of text snippets, one vector per snippet, each embedded in isolation.

Model: allenai/OLMo-2-0425-1B (hidden size 2048). Picked over the original
allenai/Olmo-3-1025-7B because the realtime installation pipeline
(convo_recorder/) has to embed live on an 8GB Apple M2 with no discrete GPU,
where the 7B model (~14GB in fp16) doesn't fit. The bank
(data/utterance_library/aggregated_segments_with_embeddings.csv, see
reembed_utterance_bank.py) and the realtime query path both go through this
same module so they can never drift into different embedding spaces - only
self-consistency matters here, not matching any particular "true" embedding.

Two backends:
- "mlx" (default): mlx-lm, confirmed faster on this hardware (see
  analysis/scripts/benchmark_olmo_backend.py: ~960ms/utterance vs ~1190ms,
  19s load once cached vs 107s) and the one the realtime pipeline uses.
  Embeds one text at a time - batch_size is ignored. There's no batching
  win to chase: realtime usage is inherently one utterance at a time, and
  the one place that embeds many texts at once (the offline bank re-embed)
  is a one-time background job where wall-clock time doesn't matter.
- "transformers": the original implementation, kept as a fallback/reference
  path (e.g. if a future model isn't mlx-lm-supported yet). Batches for
  throughput.

OLMo is a decoder-only LM with no CLS token, so the whole-snippet vector is
the last token's hidden state (pooling="last"); "mean" averages over the
snippet's tokens instead.

Run in the `artsinteg` env (has mlx, mlx-lm, transformers, torch). Usage:
    from text_embeddings import add_embeddings
    df = add_embeddings(df)                  # adds emb_0 .. emb_2047
"""

import gc

import numpy as np
import pandas as pd
from tqdm import tqdm

DEFAULT_MODEL = "allenai/OLMo-2-0425-1B"

# mlx-lm compiles (and caches) a distinct kernel graph per unique input shape.
# Utterances are all different lengths, so embedding them one at a time at
# their exact length made mlx compile thousands of never-reused graphs -
# ~20s/utterance and multiple GB of compressed/swapped memory on an 8GB
# machine, instead of the ~1s benchmarked on a handful of samples. Padding
# every input up to the next bucket keeps the shape count small (len(buckets)
# + 1) no matter how many utterances are embedded, so graphs get reused
# instead of piling up. 512 tokens covers any realistic utterance/segment;
# anything longer falls through to its exact (rare, one-off) length.
MLX_LENGTH_BUCKETS = (8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)


class TextEmbedder:
    """Loads OLMo once; embed() can then be called on any number of text lists."""

    def __init__(self, model_name=DEFAULT_MODEL, device=None, dtype="float16", layer=-1,
                 backend="mlx"):
        if backend not in ("mlx", "transformers"):
            raise ValueError(f"backend must be 'mlx' or 'transformers', not {backend!r}")
        if backend == "mlx" and layer != -1:
            raise ValueError("mlx backend only exposes the final hidden state (layer=-1); "
                              "use backend='transformers' for other layers")

        self.backend = backend
        self.model_name = model_name
        self.layer = layer
        self.hidden_size = None  # mlx: discovered on first embed() call

        if backend == "mlx":
            import mlx.core as mx
            from mlx_lm import load
            self.mx = mx
            self.model, self.tokenizer = load(model_name)
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.torch = torch
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=getattr(torch, dtype)).to(self.device).eval()
            self.hidden_size = self.model.config.hidden_size

    def embed(self, texts, pooling="last", prompt=None, batch_size=16) -> np.ndarray:
        """Embed each text on its own -> float32 array of shape (len(texts), hidden_size).

        prompt, if given, is prepended to every text; "mean" pooling then averages
        only the text's tokens, since the shared prompt would pull every vector
        toward a constant. batch_size is ignored by the mlx backend (see module
        docstring).
        """
        if pooling not in ("last", "mean"):
            raise ValueError(f"pooling must be 'last' or 'mean', not {pooling!r}")
        if self.backend == "mlx":
            return self._embed_mlx(texts, pooling, prompt)
        return self._embed_transformers(texts, pooling, prompt, batch_size)

    def _embed_mlx(self, texts, pooling, prompt) -> np.ndarray:
        mx = self.mx
        encode = lambda s: self.tokenizer.encode(s, add_special_tokens=False)
        prompt_ids = encode(prompt) if prompt else []
        start = len(prompt_ids)
        pad_id = self.tokenizer.pad_token_id or 0

        vecs = []
        for text in tqdm(texts, desc="embedding", leave=False):
            ids = prompt_ids + encode(text)
            length = len(ids)
            bucket = next((b for b in MLX_LENGTH_BUCKETS if b >= length), length)
            padded = ids + [pad_id] * (bucket - length)
            arr = mx.array([padded])
            # model.model = embed_tokens -> layers -> final norm, i.e. the bare
            # transformer without the lm_head projection - the mlx-lm analogue
            # of transformers' output_hidden_states[-1]. Right-padding is safe
            # under causal attention (real tokens never see the pads after
            # them), same reasoning as the transformers path below - we just
            # read the real last-token position instead of array position -1.
            hidden = self.model.model(arr)
            mx.eval(hidden)
            if self.hidden_size is None:
                self.hidden_size = hidden.shape[-1]
            if pooling == "last":
                vec = hidden[0, length - 1, :]
            else:
                vec = hidden[0, start:length, :].mean(axis=0)
            vecs.append(np.array(vec, dtype=np.float32))
        return np.stack(vecs)

    def _embed_transformers(self, texts, pooling, prompt, batch_size) -> np.ndarray:
        torch = self.torch
        encode = lambda s: self.tokenizer.encode(s, add_special_tokens=False)
        prompt_ids = encode(prompt) if prompt else []
        ids = [prompt_ids + encode(t) for t in texts]
        start = len(prompt_ids)

        # Longest first: minimizes padding, and any OOM happens on the first batch.
        order = sorted(range(len(ids)), key=lambda i: -len(ids[i]))
        out = np.empty((len(ids), self.hidden_size), dtype=np.float32)
        pad = self.tokenizer.pad_token_id
        for b in tqdm(range(0, len(order), batch_size), desc="embedding", leave=False):
            batch = order[b:b + batch_size]
            lengths = torch.tensor([len(ids[i]) for i in batch])
            # Right padding: under causal attention, real tokens never see the pads,
            # so each vector matches embedding that text alone.
            input_ids = torch.full((len(batch), int(lengths.max())), pad, dtype=torch.long)
            for row, i in enumerate(batch):
                input_ids[row, :len(ids[i])] = torch.tensor(ids[i])
            positions = torch.arange(input_ids.shape[1])
            mask = (positions[None, :] < lengths[:, None]).long()

            with torch.no_grad():
                hidden = self.model(input_ids=input_ids.to(self.device),
                                    attention_mask=mask.to(self.device),
                                    output_hidden_states=True, use_cache=False
                                    ).hidden_states[self.layer].float().cpu()
            if pooling == "last":
                vecs = hidden[torch.arange(len(batch)), lengths - 1]
            else:
                keep = (mask.bool() & (positions[None, :] >= start)).float()[..., None]
                vecs = (hidden * keep).sum(1) / keep.sum(1)
            out[batch] = vecs.numpy()
        return out

    def unload(self):
        del self.model, self.tokenizer
        gc.collect()
        if self.backend == "transformers" and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def add_embeddings(df: pd.DataFrame, text_col="text", prefix="emb_", embedder=None,
                   **embed_kwargs) -> pd.DataFrame:
    """Return df with one embedding column per dimension ({prefix}0 .. {prefix}N-1).

    Each row's text is embedded in isolation. Pass an existing TextEmbedder to reuse a
    loaded model; otherwise one is loaded for this call and freed afterwards.
    embed_kwargs go to TextEmbedder.embed (pooling, prompt, batch_size).
    """
    texts = df[text_col]
    bad = texts.map(lambda t: not isinstance(t, str) or not t.strip())
    if bad.any():
        raise ValueError(f"{bad.sum()} rows have empty or non-string {text_col!r}; drop them first")

    own = embedder is None
    if own:
        embedder = TextEmbedder()
    try:
        vecs = embedder.embed(texts.str.strip().tolist(), **embed_kwargs)
    finally:
        if own:
            embedder.unload()

    emb = pd.DataFrame(vecs, index=df.index, columns=[f"{prefix}{i}" for i in range(vecs.shape[1])])
    return pd.concat([df, emb], axis=1)
