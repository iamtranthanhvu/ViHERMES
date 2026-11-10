"""LoRA fine-tuning of the Conductor on chat samples from ``make-lora-data``.

The prompt is built with the model's own chat template exactly as at inference time (thinking
switched off), and only the answer tokens carry loss. The best epoch by validation loss is kept as a
small adapter under ``output/adapter``. Heavy libraries load only when training starts.
"""

import json
import time
from pathlib import Path


def encode_example(tokenizer, messages, max_seq_length, no_thinking=True):
    """Token ids and labels for one [system, user, assistant] sample; None if it is too long.

    Labels are -100 (ignored) over the prompt and equal the ids over the answer plus the end
    token, so the model learns the answer and when to stop.
    """
    options = {"enable_thinking": False} if no_thinking else {}
    prompt = tokenizer.apply_chat_template(
        messages[:2], add_generation_prompt=True, tokenize=False, **options
    )
    answer = messages[2]["content"] + (tokenizer.eos_token or "")
    # tokenize=False plus an explicit call: apply_chat_template(tokenize=True) returns a list in
    # transformers 4.x but a BatchEncoding in 5.x.
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if len(prompt_ids) + len(answer_ids) > max_seq_length:
        return None
    return {"input_ids": prompt_ids + answer_ids, "labels": [-100] * len(prompt_ids) + answer_ids}


def collate(batch, pad_id):
    """Right-pad a batch; padded labels are -100 so they carry no loss."""
    import torch

    width = max(len(item["input_ids"]) for item in batch)
    pad = lambda values, fill: values + [fill] * (width - len(values))  # noqa: E731
    return {
        "input_ids": torch.tensor([pad(i["input_ids"], pad_id) for i in batch]),
        "labels": torch.tensor([pad(i["labels"], -100) for i in batch]),
        "attention_mask": torch.tensor([pad([1] * len(i["input_ids"]), 0) for i in batch]),
    }


def _check_not_forbidden(rows, qids, passages):
    for part, items in rows.items():
        for row in items:
            qid = row["meta"]["qid"]
            if qid in qids:
                raise ValueError(f"forbidden question {qid} found in the {part} rows")
            try:
                payload = json.loads(row["messages"][1]["content"])
            except ValueError:
                raise ValueError(f"Row {qid} in {part}: the user message is not the JSON payload")
            shown = {e["id"] for e in payload.get("evidence", [])}
            clash = shown & set(passages)
            if clash:
                raise ValueError(f"forbidden passage {sorted(clash)[0]} found in the {part} rows")


def _versions():
    from importlib import metadata

    out = {}
    for name in ("peft", "trl", "transformers", "torch", "accelerate", "datasets"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return {"packages": out}


def _load_model(base, revision, dtype):
    import transformers
    from packaging.version import Version

    # transformers 4.56 renamed torch_dtype to dtype; older versions would silently ignore it.
    key = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
    options = {key: dtype}
    if revision:
        options["revision"] = revision
    try:
        return transformers.AutoModelForCausalLM.from_pretrained(base, **options)
    except (ValueError, KeyError):
        # Some multimodal checkpoints are not registered for text-only loading.
        return transformers.AutoModelForImageTextToText.from_pretrained(base, **options)


class _Rows:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def train_lora(
    train_rows,
    validation_rows,
    output,
    base="Qwen/Qwen3.5-4B",
    revision=None,
    epochs=2,
    lr=2e-4,
    batch_size=1,
    grad_accum=8,
    max_seq_length=8192,
    rank=16,
    alpha=32,
    dropout=0.05,
    max_steps=-1,
    seed=42,
    no_thinking=True,
    gradient_checkpointing=True,
    forbidden_qids=(),
    forbidden_passages=(),
    meta=None,
):
    """Train a LoRA adapter; keep the epoch with the lowest validation loss; return the run record."""
    import dataclasses

    import torch
    import transformers
    from peft import LoraConfig, get_peft_model

    _check_not_forbidden(
        {"train": train_rows, "validation": validation_rows},
        set(forbidden_qids),
        set(forbidden_passages),
    )
    output = Path(output)
    adapter_dir = output / "adapter"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base, **({"revision": revision} if revision else {})
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded, dropped, lengths = {}, {}, []
    for name, items in (("train", train_rows), ("validation", validation_rows)):
        kept = [
            encode_example(tokenizer, r["messages"], max_seq_length, no_thinking) for r in items
        ]
        dropped[name] = sum(e is None for e in kept)
        encoded[name] = [e for e in kept if e is not None]
        lengths += [len(e["input_ids"]) for e in encoded[name]]
    if not encoded["train"]:
        raise ValueError(f"No training row fits max_seq_length={max_seq_length} tokens")

    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    model = _load_model(base, revision, torch.bfloat16 if bf16 else torch.float32)
    if gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()  # needed for checkpointing with frozen base weights
    model = get_peft_model(
        model,
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    history = {"epochs": [], "best": None, "best_epoch": None}

    class KeepBestOnValidation(transformers.TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
            loss = (metrics or {}).get("eval_loss")
            label = f"epoch {state.epoch:.2f}" if state.epoch is not None else "final"
            history["epochs"].append(
                {"label": label, "step": state.global_step, "validation_loss": loss}
            )
            if loss is not None and (history["best"] is None or loss < history["best"]):
                history["best"], history["best_epoch"] = loss, label
                model.save_pretrained(str(adapter_dir))

    fields = {f.name for f in dataclasses.fields(transformers.TrainingArguments)}
    options = {
        "output_dir": str(output / "checkpoints"),
        "num_train_epochs": epochs,
        "max_steps": max_steps,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": lr,
        "lr_scheduler_type": "cosine",
        "bf16": bf16,
        "use_cpu": not cuda,
        "seed": seed,
        "logging_steps": 5,
        "save_strategy": "no",
        "eval_strategy": "epoch" if encoded["validation"] else "no",
        "report_to": "none",
        "remove_unused_columns": False,
        ("warmup_ratio" if "warmup_ratio" in fields else "warmup_steps"): 0.03,
    }
    arguments = transformers.TrainingArguments(**options)
    pad_id = tokenizer.pad_token_id
    trainer = transformers.Trainer(
        model=model,
        args=arguments,
        train_dataset=_Rows(encoded["train"]),
        eval_dataset=_Rows(encoded["validation"]) if encoded["validation"] else None,
        data_collator=lambda batch: collate(batch, pad_id),
        callbacks=[KeepBestOnValidation()],
    )
    started = time.perf_counter()
    trainer.train()
    if (
        encoded["validation"] and history["best"] is None
    ):  # e.g. max_steps ended before an epoch did
        trainer.evaluate()
    if not adapter_dir.exists():
        model.save_pretrained(str(adapter_dir))
    seconds = time.perf_counter() - started

    ordered = sorted(lengths)
    record = {
        "base": base,
        "revision": revision,
        "seed": seed,
        "rows": {k: len(v) for k, v in encoded.items()},
        "dropped_too_long": dropped,
        "tokens": {
            "p50": ordered[len(ordered) // 2],
            "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
            "max": ordered[-1],
        },
        "steps": trainer.state.global_step,
        "params": {
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "max_seq_length": max_seq_length,
            "rank": rank,
            "alpha": alpha,
            "dropout": dropout,
            "max_steps": max_steps,
            "no_thinking": no_thinking,
            "gradient_checkpointing": gradient_checkpointing,
        },
        "trainable_params": trainable,
        "total_params": total,
        "validation_loss": history,
        "loss": [
            {"step": e["step"], "loss": e["loss"]} for e in trainer.state.log_history if "loss" in e
        ],
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if cuda else None,
        "seconds": seconds,
        **_versions(),
        **(meta or {}),
    }
    (output / "train_meta.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return record
