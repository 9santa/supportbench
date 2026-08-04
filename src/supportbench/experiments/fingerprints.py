import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class GitState:
    commit: str
    branch: str
    dirty: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def read_git_state(
    project_root: Path,
) -> GitState:
    return GitState(
        commit=_git_output(
            project_root,
            "rev-parse",
            "HEAD",
        ),
        branch=_git_output(
            project_root,
            "rev-parse",
            "--abbrev-ref",
            "HEAD",
        ),
        dirty=bool(
            _git_output(
                project_root,
                "status",
                "--porcelain",
            )
        ),
    )


def _git_output(project_root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown error"

    return result.stdout.strip()
