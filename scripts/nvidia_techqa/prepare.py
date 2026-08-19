import argparse
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from huggingface_hub import hf_hub_download

from scripts._paths import PROJECT_ROOT
from supportbench.applications.nvidia_techqa import (
    DEFAULT_CHUNK_CONFIG,
    DEFAULT_DENSE_MODEL,
)
from supportbench.chunking import (
    HeadingAwareChunker,
    HuggingFaceTokenCodec,
    build_chunk_corpus,
)
from supportbench.corpus.nvidia_techqa import (
    prepare_nvidia_techqa,
)
from supportbench.data.loaders import load_documents
from supportbench.retrieval.dense_build import (
    build_dense_index,
)
from supportbench.retrieval.dense_encoder import (
    SentenceTransformerDenceEncoder,
)


DATASET_ID = "nvidia/TechQA-RAG-Eval"

# HF dataset commit used by the project.
DATASET_REVISION = "0b5bbc84b7f07d6d09d063130e90b716d8d4a32a"

DATA_ROOT = PROJECT_ROOT / "data" / "nvidia_techqa"
RAW_ROOT = DATA_ROOT / "raw"
NORMALIZED_ROOT = DATA_ROOT / "normalized"

CHUNK_DIR = DATA_ROOT / "chunks" / DEFAULT_CHUNK_CONFIG

INDEX_DIR = PROJECT_ROOT / "artifacts" / "nvidia_techqa" / "indexes" / DEFAULT_CHUNK_CONFIG


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dense-device",
        default="cuda",
        choices=("cpu", "cuda"),
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    dataset_zip, corpus_zip = download_sources(force=args.force)

    prepare_corpus(
        dataset_zip=dataset_zip,
        corpus_zip=corpus_zip,
        force=args.force,
    )

    prepare_chunks(force=args.force)

    prepare_index(
        device=args.dense_device,
        force=args.force,
    )

    validate()

    print()
    print("TechQA resources are ready.")
    print(f"Chunks: {CHUNK_DIR}")
    print(f"Index:  {INDEX_DIR}")


def download_sources(
    *,
    force: bool,
) -> tuple[Path, Path]:
    RAW_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = RAW_ROOT / "train.json"
    corpus_path = RAW_ROOT / "corpus.zip"
    dataset_zip = RAW_ROOT / "dataset.zip"

    if force or not train_path.exists():
        downloaded = hf_hub_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            revision=DATASET_REVISION,
            filename="train.json",
        )

        shutil.copyfile(
            downloaded,
            train_path,
        )

    if force or not corpus_path.exists():
        downloaded = hf_hub_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            revision=DATASET_REVISION,
            filename="corpus.zip",
        )

        shutil.copyfile(
            downloaded,
            corpus_path,
        )

    # Existing normalizer expects a ZIP containing train.json
    if force or not dataset_zip.exists():
        with ZipFile(
            dataset_zip,
            "w",
            compression=ZIP_DEFLATED,
        ) as archive:
            archive.write(
                train_path,
                arcname="train.json",
            )

    return dataset_zip, corpus_path


def prepare_corpus(
    *,
    dataset_zip: Path,
    corpus_zip: Path,
    force: bool,
) -> None:
    documents_path = NORMALIZED_ROOT / "documents.jsonl"

    if documents_path.exists() and not force:
        print("Normalized corpus exists, skipping.")
        return

    if force:
        shutil.rmtree(
            NORMALIZED_ROOT,
            ignore_errors=True,
        )

    print("Preparing normalized TechQA corpus...")

    prepare_nvidia_techqa(
        dataset_zip=dataset_zip,
        corpus_zip=corpus_zip,
        output_dir=NORMALIZED_ROOT,
    )


def prepare_chunks(
    *,
    force: bool,
) -> None:
    chunks_path = CHUNK_DIR / "chunks.jsonl"

    if chunks_path.exists() and not force:
        print("Chunks exist, skipping.")
        return

    if force:
        shutil.rmtree(
            CHUNK_DIR,
            ignore_errors=True,
        )

    print("Building heading-aware chunks...")

    documents_path = NORMALIZED_ROOT / "documents.jsonl"

    documents = load_documents(documents_path)

    token_codec = HuggingFaceTokenCodec.from_pretrained(DEFAULT_DENSE_MODEL)

    chunker = HeadingAwareChunker(
        token_codec=token_codec,
        target_tokens=384,
        oversized_overlap=64,
        max_input_tokens=512,
        special_token_reserve=2,
    )

    if chunker.chunking_key != DEFAULT_CHUNK_CONFIG:
        raise RuntimeError(f"chunk configuration does not match {DEFAULT_CHUNK_CONFIG}")

    build_chunk_corpus(
        documents=documents,
        chunker=chunker,
        token_codec=token_codec,
        tokenizer_name=DEFAULT_DENSE_MODEL,
        source_documents_path=(documents_path),
        output_directory=CHUNK_DIR,
        max_input_tokens=512,
        special_token_reserve=2,
    )


def prepare_index(
    *,
    device: str,
    force: bool,
) -> None:
    manifest_path = INDEX_DIR / "manifest.json"

    if manifest_path.exists() and not force:
        print("Dense index exists, skipping.")
        return

    if force:
        shutil.rmtree(
            INDEX_DIR,
            ignore_errors=True,
        )

    print("Building dense index...")

    documents = load_documents(CHUNK_DIR / "documents.jsonl")

    encoder = SentenceTransformerDenceEncoder(
        DEFAULT_DENSE_MODEL,
        device=device,
        batch_size=16,
    )

    build_dense_index(
        documents=documents,
        encoder=encoder,
        model_name=DEFAULT_DENSE_MODEL,
        output_directory=INDEX_DIR,
    )


def validate() -> None:
    required = (
        NORMALIZED_ROOT / "documents.jsonl",
        CHUNK_DIR / "documents.jsonl",
        CHUNK_DIR / "chunks.jsonl",
        INDEX_DIR / "manifest.json",
        INDEX_DIR / "index.faiss",
    )

    missing = [path for path in required if not path.exists()]

    if missing:
        lines = "\n".join(f"- {path}" for path in missing)

        raise RuntimeError("TechQA preparation incomplete:\n" + lines)


if __name__ == "__main__":
    main()
