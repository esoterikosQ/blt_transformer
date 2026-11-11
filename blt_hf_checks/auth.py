"""Read only a project-scoped HF token. Never execute a credentials file."""
import re
from pathlib import Path


def read_project_token(path: Path | str, project_root: Path | str) -> str:
    path, root = Path(path).resolve(), Path(project_root).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Credential file must be inside the project")
    if not path.is_file():
        raise ValueError("Project credential file is missing")
    if path.stat().st_size > 16384:
        raise ValueError("Project credential file is unexpectedly large")
    tokens = set(re.findall(r"(?<![A-Za-z0-9_])hf_[A-Za-z0-9]{10,}(?![A-Za-z0-9_])", path.read_text()))
    if len(tokens) != 1:
        raise ValueError("Expected exactly one HF access token in the project credential file; login passwords are not used")
    return tokens.pop()
