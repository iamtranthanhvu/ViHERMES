#!/usr/bin/env python3
"""Build a tiny random causal language model with a chat template, offline.

Stands in for Qwen3.5 in tests and dry runs of the LoRA pipeline: a 1-layer Llama with a
character-level vocabulary and a ChatML-style template that honours `enable_thinking`.

    python scripts/make_tiny_lm.py --output outputs/dryrun/tiny-lm
"""

import argparse
import string
from pathlib import Path

CHARS = string.ascii_lowercase + string.digits
TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and enable_thinking == false %}nothink\n{% endif %}{% endif %}"
)


def build_tiny_lm(output):
    import transformers

    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *CHARS, *("##" + c for c in CHARS)]
    (root / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tokenizer = transformers.BertTokenizerFast(
        vocab_file=str(root / "vocab.txt"), do_lower_case=True, strip_accents=True
    )
    tokenizer.chat_template = TEMPLATE
    tokenizer.pad_token, tokenizer.eos_token = "[PAD]", "[SEP]"
    tokenizer.save_pretrained(root)
    config = transformers.LlamaConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=8192,
        pad_token_id=0,
        eos_token_id=3,
    )
    transformers.LlamaForCausalLM(config).save_pretrained(root)
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(build_tiny_lm(parser.parse_args().output))
