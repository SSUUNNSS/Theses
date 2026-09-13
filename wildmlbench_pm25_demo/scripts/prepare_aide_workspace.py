"""Create the exact three-file workspace exposed to the agent."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ALLOWED_FILES = ("train.csv", "test_features.csv", "TASK.md")
FORBIDDEN_NAMES = {"test_labels.csv", "baseline.py", "evaluate.py", "README.md", "tests"}


def validate_workspace(path: Path) -> None:
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"Workspace is not a real directory: {path}")
    entries = list(path.iterdir())
    names = {entry.name for entry in entries}
    if names != set(ALLOWED_FILES):
        raise ValueError(f"Workspace must contain exactly {ALLOWED_FILES}; found {sorted(names)}")
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(f"Workspace entry must be a regular file: {entry}")
    if names & FORBIDDEN_NAMES:
        raise ValueError("Forbidden benchmark assets are present in the agent workspace.")


def prepare(root: Path, destination: Path) -> None:
    source_data = root / "data" / "processed"
    source_map = {
        "train.csv": source_data / "train.csv",
        "test_features.csv": source_data / "test_features.csv",
        "TASK.md": root / "TASK.md",
    }
    destination.mkdir(parents=True, exist_ok=True)
    for entry in destination.iterdir():
        if entry.is_symlink() or entry.is_file():
            entry.unlink()
        elif entry.is_dir():
            shutil.rmtree(entry)
    for name, source in source_map.items():
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"Required source file is unavailable: {source}")
        shutil.copy2(source, destination / name)
    validate_workspace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    destination = args.destination or args.root / "aide_task"
    prepare(args.root, destination)
    print(f"Prepared isolated workspace: {destination}")


if __name__ == "__main__":
    main()
