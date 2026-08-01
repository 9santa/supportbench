from collections.abc import Sequence


class WhitespaceTokenCodec:
    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    def encode(
        self,
        text: str,
    ) -> list[int]:
        token_ids: list[int] = []

        for token in text.split():
            token_id = self._token_to_id.get(token)

            if token_id is None:
                token_id = len(self._token_to_id)
                self._token_to_id[token] = token_id
                self._id_to_token[token_id] = token

            token_ids.append(token_id)

        return token_ids

    def decode(
        self,
        token_ids: Sequence[int],
    ) -> str:
        return " ".join(self._id_to_token[token_id] for token_id in token_ids)
