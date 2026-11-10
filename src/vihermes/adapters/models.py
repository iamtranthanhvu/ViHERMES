"""OpenAI-compatible and Sentence Transformers adapters, plus an offline fixture encoder."""

import hashlib
import json
import os

import numpy as np

from ..retrieval import tokens


class HashEmbedder:
    """Deterministic lexical hashing for smoke tests, not neural semantic retrieval."""

    identity = "fixture-hash-256-v1"

    def encode(self, texts):
        result = np.zeros((len(texts), 256), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in tokens(text):
                digest = hashlib.sha256(word.encode()).digest()
                result[row, int.from_bytes(digest[:4], "little") % 256] += 1
        return result


class OpenAIEmbedder:
    def __init__(self, model="text-embedding-3-small", batch_size=64):
        from openai import OpenAI

        self.client, self.model, self.batch_size = OpenAI(), model, batch_size
        self.identity = "openai:" + str(self.client.base_url) + ":" + model

    def encode(self, texts):
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            result = self.client.embeddings.create(
                model=self.model, input=texts[start : start + self.batch_size]
            )
            vectors.extend(item.embedding for item in sorted(result.data, key=lambda x: x.index))
        return np.asarray(vectors, dtype=np.float32)


class SentenceEmbedder:
    def __init__(self, model, batch_size=32, max_seq_length=None, revision=None):
        from sentence_transformers import SentenceTransformer

        self.model = (
            SentenceTransformer(model, revision=revision)
            if revision
            else (SentenceTransformer(model))
        )
        self.identity, self.batch_size = "sentence-transformers:" + model, batch_size
        if revision:
            self.identity += f"@{revision[:12]}"
        if max_seq_length:
            # Truncation changes the vectors, so it must change the index identity too.
            self.model.max_seq_length = max_seq_length
            self.identity += f":max{max_seq_length}"

    def encode(self, texts):
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True)


NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}


class OpenAILanguageModel:
    """Chat model behind any OpenAI-compatible endpoint (OpenAI, vLLM, ...).

    ``base_url`` points one instance at its own server, so a generator and a judge can run on
    different servers in one process. ``extra_body`` carries server-specific request fields such
    as ``NO_THINKING``.
    """

    def __init__(self, model="gpt-4o-mini", temperature=0, base_url=None, extra_body=None):
        from openai import OpenAI

        if base_url:
            # Local servers usually ignore the key, but the SDK refuses to start without one.
            self.client = OpenAI(
                base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY"
            )
        else:
            self.client = OpenAI()
        self.model, self.temperature, self.extra_body = model, temperature, extra_body

    def json(self, system, payload):
        options = {"extra_body": self.extra_body} if self.extra_body else {}
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            **options,
        )
        result = json.loads(response.choices[0].message.content or "{}")
        if not isinstance(result, dict):
            raise ValueError("Expected a JSON object from the language model")
        return result
