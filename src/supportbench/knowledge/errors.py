class KnowledgeError(Exception):
    pass


class SupportDocumentNotFoundError(KnowledgeError):
    def __init__(
        self,
        *,
        document_id: str,
    ) -> None:
        self.document_id = document_id

        super().__init__(f"support document not found: {document_id!r}")


class SupportChunkNotFoundError(KnowledgeError):
    def __init__(
        self,
        *,
        document_id: str,
        chunk_id: str,
    ) -> None:
        self.document_id = document_id
        self.chunk_id = chunk_id

        super().__init__(f"chunk {chunk_id!r} was not found in document {document_id!r}")
