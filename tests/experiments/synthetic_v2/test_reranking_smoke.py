import math
from pathlib import Path

import pytest
import torch

from supportbench.data.loaders import load_documents
from supportbench.reranking.cross_encoder import SentenceTransformerCrossEncoderReranker
from supportbench.reranking.retriever import RerankingRetriever
from supportbench.retrieval.factory import RetrieverConfig, RetrieverFactory

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOCUMENTS_PATH = PROJECT_ROOT / "data" / "synthetic" / "v2" / "documents.jsonl"
DENSE_INDEX_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "synthetic"
    / "v2"
    / "dense"
    / "multilingual-e5-base"
)


@pytest.mark.smoke
def test_reranking_retriever_smoke() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the reranking smoke test")

    documents = load_documents(DOCUMENTS_PATH)
    document_ids = {document.doc_id for document in documents}

    factory = RetrieverFactory(
        documents,
        config=RetrieverConfig(
            dense_index_path=DENSE_INDEX_PATH,
            dense_model_name="intfloat/multilingual-e5-base",
            dense_device="cuda",
            dense_batch_size=16,
            bm25_weight=1.0,
            dense_weight=1.5,
            candidate_k=100,
            rrf_k=10,
        ),
    )

    reranker = SentenceTransformerCrossEncoderReranker(
        "BAAI/bge-reranker-v2-m3",
        device="cuda",
        batch_size=16,
        max_length=512,
    )

    reranking_retriever = RerankingRetriever(
        candidate_retriever=factory.create("hybrid"),
        reranker=reranker,
        documents=documents,
        candidate_k=50,
    )

    results = reranking_retriever.search(
        "не работает vpn на ubuntu",
        top_k=10,
    )

    result_doc_ids = [result.doc_id for result in results]

    assert len(results) == 10
    assert len(set(result_doc_ids)) == 10
    assert set(result_doc_ids) <= document_ids
    assert "vpn_ubuntu_troubleshooting" in result_doc_ids
    assert [result.rank for result in results] == list(range(1, 11))
    assert all(math.isfinite(result.score) for result in results)
    assert [result.score for result in results] == sorted(
        (result.score for result in results),
        reverse=True,
    )
