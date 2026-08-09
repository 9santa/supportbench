from typing import Protocol
from uuid import uuid4


class IdGenerator(Protocol):
    def new_id(self) -> str: ...


class UuidGenerator(IdGenerator):
    def new_id(self) -> str:
        return str(uuid4())
