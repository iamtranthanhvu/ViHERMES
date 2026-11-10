"""Package one run's numbers and merge several runs into the results table.

A report holds only numbers and run settings (never weights or databases), so it is small
enough to copy off a host with one command. ``compare`` refuses to build a table from runs
that were not measured under the same conditions.
"""

import json
import tarfile
from pathlib import Path

RUN_FILES = ("metrics.json", "predictions.jsonl", "config.json", "timing.json")
REQUIRED = ("metrics.json", "config.json")
BLOCKED_SUFFIXES = {
    ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".onnx", ".db", ".npz", ".npy",
}  # fmt: skip
METRICS = (("f1", "F1 ↑"), ("judge", "LLM Judge ↑"), ("recall@5", "Recall@5 ↑"))
LABELS = {
    "b1": "B1 (baseline)",
    "b2": "B2 (+ fine-tuned embedding)",
    "b3": "B3 (+ LoRA Conductor)",
    "b4": "B4 (+ both)",
}
BASELINE = "b1"


def pack(run_dir, name, output, extra=()):
    """Write ``output`` (a .tar.gz) holding the run's result files and any ``extra`` files."""
    run_dir = Path(run_dir)
    for required in REQUIRED:
        if not (run_dir / required).is_file():
            raise ValueError(f"{run_dir} has no {required}; run the evaluation first")
    files = [run_dir / n for n in RUN_FILES if (run_dir / n).is_file()]
    for path in map(Path, extra):
        if path.suffix.lower() in BLOCKED_SUFFIXES:
            raise ValueError(f"{path.name}: weights and binary data are not allowed in a report")
        if not path.is_file():
            raise ValueError(f"Extra file {path} does not exist")
        files.append(path)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as tar:
        for path in files:
            tar.add(path, arcname=f"{name}/{path.name}")
    return output


def _read(source, filename):
    """Return (name, parsed JSON or None) for a packed report or a plain run directory."""
    source = Path(source)
    if source.is_dir():
        path = source / filename
        return source.name, json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    with tarfile.open(source, "r:gz") as tar:
        names = [m.name for m in tar.getmembers() if m.isfile()]
        top = names[0].split("/")[0] if names else source.stem
        member = f"{top}/{filename}"
        if member not in names:
            return top, None
        return top, json.loads(tar.extractfile(member).read().decode("utf-8"))


def load_report(source):
    name, metrics = _read(source, "metrics.json")
    _, config = _read(source, "config.json")
    if metrics is None or config is None:
        raise ValueError(f"{source} lacks metrics.json or config.json")
    return {"name": name, "metrics": metrics, "config": config, "source": str(source)}


def _guard_values(report):
    config, metrics = report["config"], report["metrics"]
    split = config.get("split") or {}
    judge = config.get("judge") or {}
    return {
        "split": (split.get("hash"), split.get("part")),
        "corpus": config.get("corpus"),
        "judge": (judge.get("model"), judge.get("quant"), judge.get("revision")),
        "model": config.get("model"),
        "questions": metrics.get("overall", {}).get("n"),
    }


def check_comparable(reports):
    """Raise if the runs differ in anything other than the component being tested."""
    values = {r["name"]: _guard_values(r) for r in reports}
    for name, found in values.items():
        for key in ("split", "corpus", "model", "questions"):
            value = found[key]
            if value is None or (isinstance(value, tuple) and value[0] is None):
                raise ValueError(f"Report {name} does not record its {key}")
    first = reports[0]["name"]
    problems = [
        f"{key} differs: {first}={values[first][key]} vs {r['name']}={values[r['name']][key]}"
        for key in ("split", "corpus", "judge", "model", "questions")
        for r in reports[1:]
        if values[r["name"]][key] != values[first][key]
    ]
    if problems:
        raise ValueError("Runs are not comparable: " + "; ".join(problems))


def _cell(report, key, baseline):
    value = report["metrics"]["overall"].get(key)
    if value is None:
        return None, None
    base = baseline["metrics"]["overall"].get(key)
    # round first: float noise must not print as "-0.0000"
    delta = None if base is None or report is baseline else round(value - base, 4) + 0.0
    return value, delta


def _format(value, delta, bold):
    if value is None:
        return "n/a"
    text = f"{value:.4f}"
    if delta is not None:
        text += f" ({delta:+.4f})"
    return f"**{text}**" if bold else text


def compare(sources, output_dir, baseline=BASELINE):
    """Build table.md and results.json from packed reports or run directories."""
    reports = [load_report(s) for s in sources]
    base = next((r for r in reports if r["name"] == baseline), None)
    if base is None:
        raise ValueError(f"The baseline report ({baseline}) is required to compute deltas")
    check_comparable(reports)

    best = {}
    for key, _ in METRICS:
        values = [round(r["metrics"]["overall"][key], 4) for r in reports
                  if r["metrics"]["overall"].get(key) is not None]  # fmt: skip
        best[key] = max(values) if values else None

    rows, lines = (
        [],
        [
            "| Method | " + " | ".join(label for _, label in METRICS) + " |",
            "| :-- | " + " | ".join("--:" for _ in METRICS) + " |",
        ],
    )
    for report in reports:
        row = {"name": report["name"], "delta": {}, "by_hop": report["metrics"].get("by_hop")}
        cells = []
        for key, _ in METRICS:
            value, delta = _cell(report, key, base)
            row[key] = value
            if delta is not None:
                row["delta"][key] = delta
            is_best = value is not None and round(value, 4) == best[key]
            cells.append(_format(value, delta, is_best))
        row["n"] = report["metrics"]["overall"].get("n")
        rows.append(row)
        label = LABELS.get(report["name"], report["name"])
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    guard = _guard_values(base)
    split_hash, part = guard["split"]
    judge = base["config"].get("judge") or {}
    lines += [
        "",
        f"Evaluated on the held-out `{part}` set of {guard['questions']} questions "
        f"(split `{split_hash[:8]}`), identical for every row; B1 is the reference and the "
        "bracketed values are differences from it.",
        f"Judge: {judge.get('model')} ({judge.get('quant')}); generator: {guard['model']}.",
        "With this few questions the standard error is roughly ±10 points overall (±22 points "
        "per hop), so differences smaller than that may be noise.",
    ]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = {"baseline": baseline, "split": split_hash, "rows": rows}
    (out / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result
