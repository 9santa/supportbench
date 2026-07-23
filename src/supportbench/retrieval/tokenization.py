import re

TOKEN_PATTERN = re.compile(r"[a-zа-яё0-9]+")


def tokenize(text: str) -> list[str]:
    """Extracts tokens from the text."""
    return TOKEN_PATTERN.findall(text.lower())
