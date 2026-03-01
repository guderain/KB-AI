import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient

from app.core.config import get_settings


@dataclass
class EvalCase:
    question: str
    expected_source_keyword: str


def parse_sizes(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_eval_cases(file_path: str) -> list[EvalCase]:
    raw = json.loads(Path(file_path).read_text(encoding="utf-8"))
    cases: list[EvalCase] = []
    for item in raw:
        cases.append(
            EvalCase(
                question=item["question"],
                expected_source_keyword=item["expected_source_keyword"],
            )
        )
    return cases


def make_embeddings() -> OpenAIEmbeddings:
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.OPENAI_EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
        check_embedding_ctx_length=False,
        chunk_size=10,
    )


def load_markdown_docs(knowledge_dir: str):
    loader = DirectoryLoader(
        knowledge_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    return loader.load()


def drop_collection_if_exists(collection_name: str) -> None:
    settings = get_settings()
    client = MilvusClient(uri=settings.MILVUS_URI, token=settings.MILVUS_TOKEN or None)
    if client.has_collection(collection_name=collection_name):
        client.drop_collection(collection_name=collection_name)


def build_vector_store(collection_name: str, chunk_size: int, chunk_overlap: int, docs) -> tuple[Milvus, int, float]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise RuntimeError("no chunks generated")

    drop_collection_if_exists(collection_name)
    store = Milvus(
        embedding_function=make_embeddings(),
        collection_name=collection_name,
        connection_args={"uri": settings.MILVUS_URI, "token": settings.MILVUS_TOKEN},
        auto_id=True,
    )

    start = time.perf_counter()
    store.add_documents(chunks)
    elapsed = time.perf_counter() - start
    return store, len(chunks), elapsed


def evaluate(store: Milvus, cases: list[EvalCase], top_k: int) -> tuple[float, float]:
    hit_count = 0
    costs: list[float] = []

    for case in cases:
        start = time.perf_counter()
        docs = store.similarity_search(case.question, k=top_k)
        costs.append(time.perf_counter() - start)

        sources = [doc.metadata.get("source", "") for doc in docs]
        if any(case.expected_source_keyword in src for src in sources):
            hit_count += 1

    hit_at_k = hit_count / len(cases) if cases else 0.0
    avg_retrieval_ms = (sum(costs) / len(costs) * 1000) if costs else 0.0
    return hit_at_k, avg_retrieval_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B evaluate chunk size for Milvus retrieval quality.")
    parser.add_argument("--sizes", default="600,800,1000", help="comma-separated chunk sizes")
    parser.add_argument("--overlap-ratio", type=float, default=0.2, help="overlap ratio based on chunk size")
    parser.add_argument("--top-k", type=int, default=5, help="retrieval top k")
    parser.add_argument(
        "--eval-file",
        default="scripts/eval_queries.example.json",
        help="json file path with eval questions",
    )
    parser.add_argument(
        "--disable-proxy",
        action="store_true",
        help="unset HTTP(S)_PROXY/ALL_PROXY before network calls",
    )
    parser.add_argument(
        "--collection-prefix",
        default="kb_chunks_eval",
        help="Milvus collection prefix used for evaluation",
    )
    args = parser.parse_args()

    if args.disable_proxy:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            os.environ[key] = ""

    settings = get_settings()
    cases = load_eval_cases(args.eval_file)
    docs = load_markdown_docs(settings.KNOWLEDGE_BASE_DIR)
    sizes = parse_sizes(args.sizes)

    print(f"docs={len(docs)}, eval_cases={len(cases)}, sizes={sizes}, top_k={args.top_k}")

    results: list[dict] = []
    for size in sizes:
        overlap = max(1, int(size * args.overlap_ratio))
        collection_name = f"{args.collection_prefix}_{size}_{overlap}"
        print(f"\n[RUN] size={size}, overlap={overlap}, collection={collection_name}")

        store, chunks_count, index_seconds = build_vector_store(
            collection_name=collection_name,
            chunk_size=size,
            chunk_overlap=overlap,
            docs=docs,
        )
        hit_at_k, avg_retrieval_ms = evaluate(store=store, cases=cases, top_k=args.top_k)
        row = {
            "chunk_size": size,
            "chunk_overlap": overlap,
            "chunks": chunks_count,
            "index_seconds": round(index_seconds, 2),
            "hit_at_k": round(hit_at_k, 4),
            "avg_retrieval_ms": round(avg_retrieval_ms, 2),
            "collection": collection_name,
        }
        results.append(row)
        print(row)

    print("\n=== Summary ===")
    for row in results:
        print(row)

    best = sorted(results, key=lambda x: (x["hit_at_k"], -x["avg_retrieval_ms"]), reverse=True)[0]
    print("\n=== Recommended ===")
    print(best)


if __name__ == "__main__":
    main()

