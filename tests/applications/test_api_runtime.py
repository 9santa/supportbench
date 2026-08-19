from pathlib import Path

import pytest

from supportbench.applications.api_runtime import _project_root


def test_project_root_uses_environment_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "installed-project"
    monkeypatch.setenv(
        "SUPPORTBENCH_PROJECT_ROOT",
        str(project_root),
    )

    assert _project_root() == project_root.resolve()
