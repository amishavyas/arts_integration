"""OLMo embeddings of text snippets, one vector per snippet, each embedded in isolation.

Adapted from group_conversation_multimodal/analyses/static/semantic/
extract_speech_embeddings.py: same model, dtype and hidden-layer readout, so vectors
land in the same space. OLMo is a decoder-only LM with no CLS token, so the
whole-snippet vector is the last token's hidden state (pooling="last"); "mean"
averages over the snippet's tokens instead.

Run in the `fusion` env. Usage:
    from text_embeddings import add_embeddings
    df = add_embeddings(df)                  # adds emb_0 .. emb_4095
"""

import gc

import numpy as np
import pandas as pd
from tqdm import tqdm

DEFAULT_MODEL = "allenai/Olmo-3-1025-7B"  # spelling matches the local HF cache


class TextEmbedder:
    """Loads OLMo once; embed() can then be called on any number of text lists."""

    def __init__(self, model_name=DEFAULT_MODEL, device=None, dtype="float16", layer=-1):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_name = model_name
        self.layer = layer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=getattr(torch, dtype)).to(self.device).eval()
        self.hidden_size = self.model.config.hidden_size

    def embed(self, texts, pooling="last", prompt=None, batch_size=16) -> np.ndarray:
        """Embed each text on its own -> float32 array of shape (len(texts), hidden_size).

        prompt, if given, is prepended to every text; "mean" pooling then averages
        only the text's tokens, since the shared prompt would pull every vector
        toward a constant.
        """
        if pooling not in ("last", "mean"):
            raise ValueError(f"pooling must be 'last' or 'mean', not {pooling!r}")
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
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def add_embeddings(df: pd.DataFrame, text_col="text", prefix="emb_", embedder=None,
                   **embed_kwargs) -> pd.DataFrame:
    """Return df with one embedding column per dimension ({prefix}0 .. {prefix}4095).

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
