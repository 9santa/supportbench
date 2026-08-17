from typing import Protocol

from supportbench.knowledge.models import (
    SupportDocumentMatch,
    SupportDocumentRead,
)


class SupportKnowledgeService(Protocol):
    def search(
        self,
        *,
        query: str,
    ) -> tuple[SupportDocumentMatch, ...]: ...

    def read(
        self,
        *,
        document_id: str,
        chunk_ids: tuple[str, ...] | None = None,
    ) -> SupportDocumentRead: ...
