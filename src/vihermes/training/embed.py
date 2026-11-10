"""Fine-tune an embedding model on (question, gold passage, hard negatives) pairs.

The best checkpoint is picked by Recall@5 on the validation questions; test questions are never used.
Heavy libraries (torch, sentence-transformers, datasets) are imported only when training starts.
"""

import json
import time
from pathlib import Path

import numpy as np


def validation_recall_at_k(encode, validation, corpus, k=5):
    """Macro Recall@k of ``validation`` = [(question, gold_ids)] over the whole ``corpus`` {id: text}.

    ``encode(list[str]) -> array`` must return L2-normalised vectors. Questions without gold
    passages are ignored; returns None when nothing can be scored.
    """
    scored = [(q, set(gold)) for q, gold in validation if gold]
    if not scored:
        return None
    ids = list(corpus)
    passages = np.asarray(encode([corpus[i] for i in ids]), dtype=np.float32)
    queries = np.asarray(encode([q for q, _ in scored]), dtype=np.float32)
    top = np.argsort(-(queries @ passages.T), axis=1)[:, :k]
    recalls = [len({ids[j] for j in row} & gold) / len(gold) for row, (_, gold) in zip(top, scored)]
    return float(np.mean(recalls))


def _dataset(rows, negatives):
    from datasets import Dataset

    usable = [r for r in rows if r["negatives"]]
    columns = {"anchor": [r["query"] for r in usable], "positive": [r["positive"] for r in usable]}
    for k in range(negatives):
        # Pad short rows by repeating their last negative so every row has the same columns.
        columns[f"negative_{k + 1}"] = [
            r["negatives"][min(k, len(r["negatives"]) - 1)] for r in usable
        ]
    return Dataset.from_dict(columns), len(rows) - len(usable)


def _versions():
    from importlib import metadata

    names = ("sentence-transformers", "transformers", "datasets", "torch", "accelerate")
    out = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return {"packages": out}


def train_embedding(
    rows,
    validation,
    corpus,
    output,
    base="BAAI/bge-m3",
    revision=None,
    epochs=1,
    lr=1e-5,
    batch_size=16,
    max_seq_length=512,
    max_steps=-1,
    negatives=3,
    seed=42,
    forbidden=(),
    gradient_checkpointing=False,
    meta=None,
):
    """Train, keep the best epoch on validation under ``output/model``, write ``output/train_meta.json``."""
    import torch
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from transformers import TrainerCallback

    try:  # sentence-transformers 5.x moved these; older 3.x/4.x only have the top-level paths
        from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
        from sentence_transformers.sentence_transformer.training_args import BatchSamplers
    except ImportError:
        from sentence_transformers.losses import MultipleNegativesRankingLoss
        from sentence_transformers.training_args import BatchSamplers

    used = {r["positive_id"] for r in rows} | {n for r in rows for n in r["negative_ids"]}
    clash = used & set(forbidden)
    if clash:
        raise ValueError(f"{len(clash)} forbidden (test) passages appear in the training rows")
    if not rows:
        raise ValueError("No training rows")

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model_dir = output / "model"
    cuda = torch.cuda.is_available()
    # Pick the device explicitly: sentence-transformers would choose Apple's MPS on a Mac, whose
    # backend can crash; training is meant for CUDA hosts and CPU is the safe local fallback.
    device = "cuda" if cuda else "cpu"
    options = {"revision": revision} if revision else {}
    model = SentenceTransformer(base, device=device, **options)
    model.max_seq_length = max_seq_length
    dataset, dropped = _dataset(rows, negatives)

    def recall(m):
        return validation_recall_at_k(
            lambda texts: m.encode(
                texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
            ),
            validation,
            corpus,
        )

    history = {"base": recall(model), "epochs": [], "best": None, "best_epoch": None}

    class KeepBestOnValidation(TrainerCallback):
        def _check(self, state, m, label):
            score = recall(m)
            history["epochs"].append({"label": label, "step": state.global_step, "recall@5": score})
            better = history["best"] is None or (score or 0) > (history["best"] or 0)
            if better or not model_dir.exists():
                history["best"], history["best_epoch"] = score, label
                m.save(str(model_dir))

        def on_epoch_end(self, args, state, control, model=None, **kwargs):
            self._check(state, model, f"epoch {int(round(state.epoch or 0))}")

        def on_train_end(self, args, state, control, model=None, **kwargs):
            if not history["epochs"] or history["epochs"][-1]["step"] != state.global_step:
                self._check(state, model, "final")

    arguments = SentenceTransformerTrainingArguments(
        output_dir=str(output / "checkpoints"),
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        bf16=cuda and torch.cuda.is_bf16_supported(),
        use_cpu=not cuda,
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # rows of one question share text; avoid false negatives
        gradient_checkpointing=gradient_checkpointing,
        seed=seed,
        logging_steps=10,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        loss=MultipleNegativesRankingLoss(model),
        callbacks=[KeepBestOnValidation()],
    )
    started = time.perf_counter()
    trainer.train()
    seconds = time.perf_counter() - started

    record = {
        "base": base,
        "revision": revision,
        "seed": seed,
        "pairs": len(rows),
        "pairs_without_negatives": dropped,
        "steps": trainer.state.global_step,
        "params": {
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "max_seq_length": max_seq_length,
            "negatives": negatives,
            "max_steps": max_steps,
        },
        "validation_questions": len(validation),
        "validation_recall_at_5": history,
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
