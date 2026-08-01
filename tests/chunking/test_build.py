import json
from collections.abc import Sequence
from pathlib import Path

from supportbench.chunking.build import (
    build_chunk_corpus,
)
from supportbench.chunking.fixed_token import (
    FixedTokenChunker,
)
from supportbench.data.loaders import (
    load_documents,
)
from supportbench.data.models import Document


class WhitespaceTokenCodec:
    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    def encode(
        self,
        text: str,
    ) -> list[int]:
        result: list[int] = []

        for token in text.split():
            token_id = self._token_to_id.get(token)

            if token_id is None:
                token_id = len(self._token_to_id)
                self._token_to_id[token] = token_id
                self._id_to_token[token_id] = token

            result.append(token_id)

        return result

    def decode(
        self,
        token_ids: Sequence[int],
    ) -> str:
        return " ".join(self._id_to_token[token_id] for token_id in token_ids)


def test_builds_rich_and_runtime_chunk_files(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source_documents.jsonl"
    source_path.write_text(
        '{"example": true}\n',
        encoding="utf-8",
    )

    documents = [
        Document(
            doc_id="doc-1",
            title="Example title",
            text=("zero one two three four five six seven"),
            category="support",
        )
    ]

    codec = WhitespaceTokenCodec()

    chunker = FixedTokenChunker(
        token_codec=codec,
        chunk_size=4,
        overlap=1,
    )

    output_directory = tmp_path / "chunks"

    result = build_chunk_corpus(
        documents=documents,
        chunker=chunker,
        token_codec=codec,
        tokenizer_name="fake-tokenizer",
        source_documents_path=source_path,
        output_directory=output_directory,
        max_input_tokens=20,
        special_token_reserve=2,
    )

    runtime_documents = load_documents(result.documents_path)

    assert len(runtime_documents) == 3

    assert runtime_documents[0].doc_id == ("doc-1::ft4o1::chunk_0000")
    assert runtime_documents[0].title == ("Example title")
    assert runtime_documents[0].text == ("zero one two three")

    chunk_records = [
        json.loads(line) for line in result.chunks_path.read_text(encoding="utf-8").splitlines()
    ]

    assert [record["document_id"] for record in chunk_records] == [
        "doc-1",
        "doc-1",
        "doc-1",
    ]

    assert [record["ordinal"] for record in chunk_records] == [0, 1, 2]

    assert result.statistics.document_count == 1
    assert result.statistics.total_chunks == 3
    assert result.statistics.indexable_empty_chunks == 0
