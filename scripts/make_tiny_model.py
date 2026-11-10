#!/usr/bin/env python3
"""Build a tiny random SentenceTransformer folder, offline, for dry runs and tests.

A 1-layer BERT with a character-level vocabulary: it trains and encodes in seconds on a CPU and
needs no download, so the fine-tuning pipeline can be rehearsed without a real model.

    python scripts/make_tiny_model.py --output outputs/dryrun/tiny-st
"""

import argparse
import string
import tempfile
from pathlib import Path

CHARS = string.ascii_lowercase + string.digits


def build_tiny_bert(root):
    import transformers

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *CHARS, *("##" + c for c in CHARS)]
    vocab_file = root / "vocab.txt"
    vocab_file.write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tokenizer = transformers.BertTokenizerFast(
        vocab_file=str(vocab_file), do_lower_case=True, strip_accents=True
    )
    config = transformers.BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
    )
    transformers.BertModel(config).save_pretrained(root)
    tokenizer.save_pretrained(root)
    return root


def build_tiny_st(output):
    """Wrap the tiny BERT as a SentenceTransformer (the library adds mean pooling itself)."""
    from sentence_transformers import SentenceTransformer

    with tempfile.TemporaryDirectory() as scratch:
        bert = build_tiny_bert(scratch)
        model = SentenceTransformer(str(bert), device="cpu")  # never auto-pick MPS on a Mac
        model.max_seq_length = 64
        Path(output).mkdir(parents=True, exist_ok=True)
        model.save(str(output))
    return Path(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(build_tiny_st(parser.parse_args().output))
