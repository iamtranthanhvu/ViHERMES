"""One command surface for ingestion, retrieval, QA, evaluation, and dataset construction."""

import argparse
import json
import os
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

from . import __version__
from .agents import Auditor, Conductor, Interpreter, Pathfinder
from .data import extract_contexts, load_qa
from .evaluation import evaluate, run_environment
from .graph import SQLiteGraph
from .ingestion import ingest
from .models import Plan
from .retrieval import GraphPropagator, HybridRetriever, NumpyDenseIndex, fingerprint, units


def emit(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


def embedder(args):
    from .adapters.models import HashEmbedder, OpenAIEmbedder, SentenceEmbedder

    if args.embedding == "hash":
        return HashEmbedder()
    if args.embedding == "openai":
        return OpenAIEmbedder(args.embedding_model or "text-embedding-3-small")
    if not args.embedding_model:
        raise ValueError("--embedding-model is required for sentence-transformers")
    return SentenceEmbedder(
        args.embedding_model,
        max_seq_length=args.max_seq_length,
        revision=args.embedding_revision,
    )


def graph(args, stack):
    if args.graph == "neo4j":
        from .adapters.neo4j import Neo4jGraph

        store = Neo4jGraph(
            os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            os.environ.get("NEO4J_USER", "neo4j"),
            os.environ["NEO4J_PASSWORD"],
        )
    else:
        store = SQLiteGraph(args.db)
    stack.callback(store.close)
    return store


def dense_index(args, store, stack):
    encoder = embedder(args)
    if args.vector == "milvus":
        from .adapters.milvus import MilvusDenseIndex

        index = MilvusDenseIndex(
            units(store),
            encoder,
            os.environ.get("MILVUS_URI", "http://localhost:19530"),
            os.environ.get("MILVUS_TOKEN", ""),
        )
        stack.callback(index.close)
        return index
    return NumpyDenseIndex(units(store), encoder, Path(args.index))


def pipeline(args, store, stack):
    weight = args.dense_weight
    if not 0 <= weight <= 1 or args.k < 1:
        raise ValueError("Use a dense weight in [0,1] and a positive seed count")
    dense = dense_index(args, store, stack) if weight > 0 else None
    retrieval = HybridRetriever(store, dense, weight)
    pathfinder = Pathfinder(
        retrieval, GraphPropagator(store, args.max_depth, args.max_nodes, args.max_chars), args.k
    )
    return pathfinder


def language_model(model, base_url, no_thinking):
    from .adapters.models import NO_THINKING, OpenAILanguageModel

    return OpenAILanguageModel(
        model, base_url=base_url, extra_body=NO_THINKING if no_thinking else None
    )


def conductor(args, pathfinder):
    if args.generator != "openai":
        return Conductor(Interpreter(None), pathfinder, Auditor(None), None)
    llm = language_model(args.model, args.base_url, args.no_thinking)
    # The Conductor may use a different model (e.g. a LoRA adapter served by the same endpoint);
    # the Interpreter and Auditor keep the base model.
    writer = (
        language_model(args.conductor_model, args.base_url, args.no_thinking)
        if args.conductor_model
        else llm
    )
    return Conductor(Interpreter(llm), pathfinder, Auditor(llm), writer)


def parser():
    p = argparse.ArgumentParser(prog="vihermes", description=__doc__)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("ingest", "index", "retrieve", "ask", "evaluate", "construct", "make-embed-data"):
        command = sub.add_parser(name)
        command.add_argument("--graph", choices=["sqlite", "neo4j"], default="sqlite")
        command.add_argument("--db", default="outputs/vihermes.db")
        if name in ("index", "retrieve", "ask", "evaluate", "construct", "make-embed-data"):
            command.add_argument(
                "--embedding", choices=["openai", "sentence-transformers", "hash"], default="openai"
            )
            command.add_argument("--embedding-model")
            command.add_argument("--max-seq-length", type=int)
            command.add_argument(
                "--embedding-revision", help="commit hash of the embedding model to load"
            )
            command.add_argument("--index", default="outputs/dense.npz")
            command.add_argument("--vector", choices=["numpy", "milvus"], default="numpy")
        if name in ("retrieve", "ask", "evaluate", "make-embed-data"):
            command.add_argument("--dense-weight", type=float, default=0.5)
            command.add_argument("-k", type=int, default=5)
            command.add_argument("--max-depth", type=int, default=4)
            command.add_argument("--max-nodes", type=int, default=24)
            command.add_argument("--max-chars", type=int, default=16000)
        if name in ("ask", "evaluate", "construct"):
            command.add_argument("--model", default="gpt-4o-mini")
            command.add_argument("--base-url", help="OpenAI-compatible endpoint for --model")
            command.add_argument(
                "--no-thinking", action="store_true", help="disable reasoning mode (vLLM/Qwen)"
            )
        if name in ("ask", "evaluate"):
            command.add_argument(
                "--conductor-model", help="model for the Conductor only (default: --model)"
            )
        if name in ("ask", "evaluate"):
            command.add_argument("--generator", choices=["openai", "extractive"], default="openai")
        if name == "ingest":
            command.add_argument("--input", required=True)
        elif name == "index":
            command.add_argument("--force", action="store_true")
        elif name in ("retrieve", "ask"):
            command.add_argument("question")
        elif name == "evaluate":
            command.add_argument("--qa", required=True)
            command.add_argument("--output", required=True)
            command.add_argument("--limit", type=int)
            command.add_argument("--split", help="split JSON made by `vihermes split`")
            command.add_argument("--part", choices=["train", "dev", "test"], default="test")
            command.add_argument("--resume", action="store_true")
            command.add_argument("--answers", action="store_true")
            command.add_argument("--judge", action="store_true")
            command.add_argument("--judge-model", default="gpt-4o")
            command.add_argument(
                "--judge-base-url", help="OpenAI-compatible endpoint for the judge"
            )
            command.add_argument("--ks", default="1,5,10")
        elif name == "make-embed-data":
            command.add_argument("--qa", required=True)
            command.add_argument("--split", required=True)
            command.add_argument("--output", required=True)
            command.add_argument("--negatives", type=int, default=3)
            command.add_argument("--pool", type=int, default=30)
            command.add_argument("--seed", type=int, default=42)
        elif name == "construct":
            command.add_argument("--output", required=True)
            command.add_argument("--hops", default="1,2,3,4,5")
            command.add_argument("--per-hop", type=int, default=10)
            command.add_argument("--clusters", type=int, default=20)
            command.add_argument("--per-cluster", type=int, default=3)
            command.add_argument("--overlap", type=float, default=0.5)
            command.add_argument("--test-fraction", type=float, default=0.2)
            command.add_argument("--seed", type=int, default=42)
    split = sub.add_parser("split")
    split.add_argument("--qa", required=True)
    split.add_argument("--output")
    split.add_argument("--check", help="verify this split file against --qa instead of writing")
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--test-size", type=int, default=100)
    split.add_argument("--dev", type=float, default=0.12)
    pack = sub.add_parser("pack", help="bundle one run's numbers into a report archive")
    pack.add_argument("--run", required=True)
    pack.add_argument("--name", required=True)
    pack.add_argument("--output", required=True)
    pack.add_argument("--extra", nargs="*", default=[])
    comp = sub.add_parser("compare", help="build the results table from several reports")
    comp.add_argument("reports", nargs="+")
    comp.add_argument("--output-dir", default="outputs/compare")
    comp.add_argument("--baseline", default="b1")
    ft = sub.add_parser("finetune-embed", help="fine-tune the embedding model on train pairs")
    ft.add_argument("--data", required=True, help="JSONL from make-embed-data")
    ft.add_argument("--qa", required=True)
    ft.add_argument("--split", required=True)
    ft.add_argument("--db", default="outputs/vihermes.db")
    ft.add_argument("--base", default="BAAI/bge-m3")
    ft.add_argument("--revision")
    ft.add_argument("--output", required=True)
    ft.add_argument("--epochs", type=int, default=1)
    ft.add_argument("--lr", type=float, default=1e-5)
    ft.add_argument("--batch-size", type=int, default=16)
    ft.add_argument("--max-seq-length", type=int, default=512)
    ft.add_argument("--max-steps", type=int, default=-1)
    ft.add_argument("--negatives", type=int, default=3)
    ft.add_argument("--seed", type=int, default=42)
    ft.add_argument("--gradient-checkpointing", action="store_true")
    publish = sub.add_parser("publish", help="upload a model folder to a private HF repo")
    publish.add_argument("--path", required=True)
    publish.add_argument("--repo", required=True)
    publish.add_argument("--meta", help="JSON file that receives the repo and commit hash")
    fetch = sub.add_parser("fetch", help="download a model pinned by commit hash")
    fetch.add_argument("--repo", required=True)
    fetch.add_argument("--revision", required=True)
    fetch.add_argument("--dest", required=True)
    audit = sub.add_parser("audit-data")
    audit.add_argument("--qa", required=True)
    extract = sub.add_parser("extract-contexts")
    extract.add_argument("--qa", required=True)
    extract.add_argument("--output", required=True)
    demo = sub.add_parser("demo")
    demo.add_argument("--output", default="outputs/demo")
    return p


def demo(output):
    """A synthetic amendment example; independent of the research benchmark."""
    from .adapters.models import HashEmbedder
    from .ingestion import parse_document

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with SQLiteGraph(":memory:") as store:
        documents = [
            (
                "01/2020/TT-BYT",
                "Thông tư 01/2020/TT-BYT (ví dụ)",
                "Điều 2. Hồ sơ đăng ký\n1. Hồ sơ đăng ký gồm bản giấy.",
            ),
            (
                "02/2025/TT-BYT",
                "Thông tư 02/2025/TT-BYT (ví dụ)",
                "Điều 1. Sửa đổi Điều 2 của Thông tư số 01/2020/TT-BYT.\nHồ sơ đăng ký gồm bản điện tử.",
            ),
        ]
        for key, title, text in documents:
            nodes, edges = parse_document(key, title, text)
            for node in nodes:
                store.upsert_node(node)
            for edge in edges:
                store.upsert_edge(edge)
        dense = NumpyDenseIndex(units(store), HashEmbedder())
        pathfinder = Pathfinder(HybridRetriever(store, dense), GraphPropagator(store), k=2)
        engine = Conductor(Interpreter(), pathfinder, Auditor())
        result = engine.ask("Hồ sơ đăng ký gồm những gì theo Thông tư 01/2020/TT-BYT?")
        payload = asdict(result)
        (output / "answer.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        )
        return payload


def run(args):
    if args.command == "demo":
        return demo(args.output)
    if args.command == "extract-contexts":
        return {"contexts": extract_contexts(args.qa, args.output)}
    if args.command == "finetune-embed":
        from .graph import SQLiteGraph
        from .splits import load_split
        from .training.embed import train_embedding

        split = load_split(args.split)
        by_id = {e.id: e for e in load_qa(args.qa)}
        rows = [
            json.loads(line) for line in Path(args.data).read_text("utf-8").splitlines() if line
        ]
        test_gold = {c for i in split["test"] for c in by_id[i].gold_context_ids}
        dev = [(by_id[i].question, by_id[i].gold_context_ids) for i in split["dev"]]
        with SQLiteGraph(args.db) as store:
            corpus = {n.id: n.text for n in units(store)}
        return train_embedding(
            rows, dev, corpus, args.output, base=args.base, revision=args.revision,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
            max_seq_length=args.max_seq_length, max_steps=args.max_steps,
            negatives=args.negatives, seed=args.seed, forbidden=test_gold,
            gradient_checkpointing=args.gradient_checkpointing,
            meta={"split_hash": split["hash"], "split_file": args.split},
        )  # fmt: skip
    if args.command == "pack":
        from .report import pack

        return {"report": str(pack(args.run, args.name, args.output, args.extra))}
    if args.command == "compare":
        from .report import compare

        result = compare(args.reports, args.output_dir, args.baseline)
        return {"table": str(Path(args.output_dir) / "table.md"), **result}
    if args.command == "publish":
        from .hub import publish

        revision = publish(args.path, args.repo, args.meta)
        return {"repo": args.repo, "revision": revision}
    if args.command == "fetch":
        from .hub import fetch

        return {"path": str(fetch(args.repo, args.revision, args.dest)), "repo": args.repo}
    if args.command == "split":
        from .splits import load_split, make_split, save_split

        if args.check:
            stored = load_split(args.check)
            params = stored["params"]
            fresh = make_split(load_qa(args.qa), stored["seed"], params["test_size"], params["dev"])
            if fresh["hash"] != stored["hash"]:
                raise ValueError(
                    f"{args.check} does not match {args.qa}: the dataset or the split code changed"
                )
            return {"matches": True, "hash": stored["hash"]}
        if not args.output:
            raise ValueError("--output is required unless --check is given")
        split = make_split(load_qa(args.qa), args.seed, args.test_size, args.dev)
        save_split(split, args.output)
        return {"output": str(args.output), "hash": split["hash"], **split["stats"]}
    if args.command == "audit-data":
        examples = load_qa(args.qa)
        return {
            "records": len(examples),
            "by_hop": dict(Counter(e.hop_level for e in examples)),
            "duplicate_source_ids": len(examples)
            - len({e.metadata["original_id"] for e in examples}),
            "unique_evaluation_ids": len({e.id for e in examples}),
            "gold_hop_mismatches": sum(len(e.gold_context_ids) != e.hop_level for e in examples),
        }
    with ExitStack() as stack:
        store = graph(args, stack)
        if args.command == "ingest":
            return ingest(args.input, store)
        if args.command == "index":
            if args.force and args.vector == "numpy":
                Path(args.index).unlink(missing_ok=True)
            index = dense_index(args, store, stack)
            return {"units": len(index.ids), "vector": args.vector, "embedding": args.embedding}
        if args.command == "construct":
            from .construction import construct

            return construct(
                units(store),
                embedder(args),
                language_model(args.model, args.base_url, args.no_thinking),
                args.output,
                tuple(int(h) for h in args.hops.split(",")),
                args.per_hop,
                args.clusters,
                args.per_cluster,
                args.overlap,
                args.test_fraction,
                args.seed,
            )
        pathfinder = pipeline(args, store, stack)
        if args.command == "make-embed-data":
            from .splits import load_split
            from .training.embed_data import build_embed_data

            rows, stats = build_embed_data(
                load_qa(args.qa),
                load_split(args.split),
                pathfinder.retriever,
                {n.id: n.text for n in units(store)},
                args.negatives,
                args.pool,
                args.seed,
            )
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            return {"output": str(output), **stats}
        if args.command == "retrieve":
            return asdict(pathfinder.retrieve(Plan(args.question)))
        if args.command == "ask":
            return asdict(conductor(args, pathfinder).ask(args.question))
        examples = load_qa(args.qa)
        split_info = None
        if args.split:
            from .splits import load_split

            split = load_split(args.split)
            by_id = {e.id: e for e in examples}
            missing = [i for i in split[args.part] if i not in by_id]
            if missing:
                raise ValueError(f"Split refers to {len(missing)} questions absent from --qa")
            examples = [by_id[i] for i in split[args.part]]
            split_info = {"hash": split["hash"], "part": args.part}
        if args.limit is not None:
            if args.limit < 1:
                raise ValueError("--limit must be positive")
            examples = examples[: args.limit]
        ks = tuple(int(k) for k in args.ks.split(","))
        if not ks or min(ks) < 1:
            raise ValueError("All recall cutoffs must be positive")
        if args.judge and not args.answers:
            raise ValueError("--judge requires --answers")
        config = vars(args).copy()
        config.update(
            split_file=args.split,
            split=split_info,
            corpus=fingerprint(units(store)),
            judge={
                "model": args.judge_model,
                "quant": os.environ.get("JUDGE_QUANT") or "bf16",
                "revision": os.environ.get("JUDGE_REVISION") or None,
            },
            environment=run_environment(),
        )
        Path(args.output).mkdir(parents=True, exist_ok=True)
        (Path(args.output) / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        return evaluate(
            examples,
            pathfinder,
            args.output,
            conductor(args, pathfinder) if args.answers else None,
            language_model(args.judge_model, args.judge_base_url, args.no_thinking)
            if args.judge
            else None,
            ks,
            resume=args.resume,
            metadata={"split": split_info},
        )


def main():
    p = parser()
    args = p.parse_args()
    try:
        emit(run(args))
    except (ValueError, KeyError, ImportError, OSError) as error:
        p.exit(2, f"vihermes: {error}\n")
