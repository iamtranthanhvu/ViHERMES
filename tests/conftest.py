"""Shared fixtures. Heavy-library fixtures skip cleanly when the library is not installed."""

import importlib.util
from pathlib import Path

import pytest


def _tiny_models():
    path = Path(__file__).resolve().parent.parent / "scripts" / "make_tiny_model.py"
    spec = importlib.util.spec_from_file_location("make_tiny_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def tiny_bert_dir(tmp_path_factory):
    """A random 1-layer BERT with a character-level vocabulary, built offline in seconds."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    return _tiny_models().build_tiny_bert(tmp_path_factory.mktemp("tiny_bert"))


@pytest.fixture(scope="session")
def tiny_st_dir(tmp_path_factory):
    """The tiny BERT as a SentenceTransformer folder."""
    pytest.importorskip("sentence_transformers")
    return _tiny_models().build_tiny_st(tmp_path_factory.mktemp("tiny_st"))
