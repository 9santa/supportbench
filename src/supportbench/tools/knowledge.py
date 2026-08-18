from collections.abc import Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from supportbench.knowledge.errors import (
    SupportChunkNotFoundError,
    SupportDocumentNotFoundError,
)
from supportbench.knowledge.protocols import SupportKnowledgeService
from supportbench.tools.definitions import ToolDefinition
from supportbench.tools.exception_mapping import ToolExceptionMapper
from supportbench.tools.handlers import ToolHandler
from supportbench.tools.models import (
    ToolErrorInfo,
    ToolExecutionContext,
)


class SearchSupportDocsArguments(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    query: str = Field(
        min_length=1,
        max_length=1000,
    )


class ReadSupportDocArguments(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    document_id: str = Field(
        min_length=1,
        max_length=256,
    )

    chunk_ids: list[str] | None = Field(
        default=None,
        min_length=1,
        max_length=8,
    )

    @field_validator("chunk_ids")
    @classmethod
    def validate_chunk_ids(
        cls,
        value: list[str] | None,
    ) -> list[str] | None:
        if value is None:
            return None

        normalized = [item.strip() for item in value]

        if any(not item for item in normalized):
            raise ValueError("chunk_ids must contain non-empty strings")

        if len(set(normalized)) != len(normalized):
            raise ValueError("chunk_ids must be unique")

        return normalized


SEARCH_SUPPORT_DOCS = ToolDefinition(
    name="search_support_docs",
    description=(
        "Search technical support documentation "
        "for evidence about product behavior, "
        "requirements, compatibility, errors, "
        "configuration, fixes, or troubleshooting."
    ),
    arguments_schema=(SearchSupportDocsArguments.model_json_schema()),
    mutating=False,
)


READ_SUPPORT_DOC = ToolDefinition(
    name="read_support_doc",
    description=(
        "Read technical evidence from a "
        "support document returned by "
        "search_support_docs. Optional chunk_ids "
        "can restrict the read to specific chunks."
    ),
    arguments_schema=(ReadSupportDocArguments.model_json_schema()),
    mutating=False,
)


class SearchSupportDocsHandler(ToolHandler):
    def __init__(
        self,
        service: SupportKnowledgeService,
    ) -> None:
        self._service = service

    @property
    def definition(self) -> ToolDefinition:
        return SEARCH_SUPPORT_DOCS

    def execute(
        self,
        *,
        call_id: str,
        arguments: Mapping[str, object],
        context: ToolExecutionContext,
    ) -> Mapping[str, object]:
        args = SearchSupportDocsArguments.model_validate(arguments)

        matches = self._service.search(query=args.query)

        return {
            "matches": [
                {
                    "document_id": match.document_id,
                    "title": match.title,
                    "rank": match.rank,
                    "evidence": [
                        {
                            "chunk_id": chunk.chunk_id,
                            "section": (
                                " > ".join(chunk.section_path) if chunk.section_path else "<root>"
                            ),
                            "text": chunk.text,
                            "truncated": (chunk.truncated),
                        }
                        for chunk in match.evidence
                    ],
                }
                for match in matches
            ]
        }


class ReadSupportDocHandler(ToolHandler):
    def __init__(
        self,
        service: SupportKnowledgeService,
    ) -> None:
        self._service = service

    @property
    def definition(self) -> ToolDefinition:
        return READ_SUPPORT_DOC

    def execute(
        self,
        *,
        call_id: str,
        arguments: Mapping[str, object],
        context: ToolExecutionContext,
    ) -> Mapping[str, object]:
        args = ReadSupportDocArguments.model_validate(arguments)

        document = self._service.read(
            document_id=args.document_id,
            chunk_ids=tuple(args.chunk_ids) if args.chunk_ids is not None else None,
        )

        return {
            "document_id": document.document_id,
            "title": document.title,
            "truncated": document.truncated,
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "section": (" > ".join(chunk.section_path) if chunk.section_path else "<root>"),
                    "text": chunk.text,
                    "truncated": chunk.truncated,
                }
                for chunk in document.chunks
            ],
        }


def build_knowledge_tool_handlers(
    service: SupportKnowledgeService,
) -> tuple[ToolHandler, ...]:
    return (
        SearchSupportDocsHandler(service),
        ReadSupportDocHandler(service),
    )


class KnowledgeToolExceptionMapper(ToolExceptionMapper):
    def map_exception(
        self,
        exc: Exception,
    ) -> ToolErrorInfo | None:
        if isinstance(
            exc,
            SupportDocumentNotFoundError,
        ):
            return ToolErrorInfo(
                code=("support_document_not_found"),
                message=("The requested support document was not found."),
            )

        if isinstance(
            exc,
            SupportChunkNotFoundError,
        ):
            return ToolErrorInfo(
                code="support_chunk_not_found",
                message=("The requested support chunk was not found in the document."),
            )

        return None
