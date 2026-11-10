"""Shared fixtures. Heavy-library fixtures skip cleanly when the library is not installed."""

import string

import pytest

CHARS = string.ascii_lowercase + string.digits


@pytest.fixture(scope="session")
def tiny_bert_dir(tmp_path_factory):
    """A random, 1-layer BERT with a character-level vocabulary, built offline in seconds."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    root = tmp_path_factory.mktemp("tiny_bert")
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


@pytest.fixture(scope="session")
def tiny_st_dir(tiny_bert_dir, tmp_path_factory):
    """The tiny BERT as a SentenceTransformer folder (the library adds mean pooling itself)."""
    st = pytest.importorskip("sentence_transformers")
    model = st.SentenceTransformer(str(tiny_bert_dir), device="cpu")
    model.max_seq_length = 64
    path = tmp_path_factory.mktemp("tiny_st")
    model.save(str(path))
    return path
