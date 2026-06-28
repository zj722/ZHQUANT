from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .dsl_validator import DSLValidationError, StrategyDSLValidator


def iter_strategy_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob("*.json"))
    raise FileNotFoundError(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate ZHQUANT strategy DSL JSON files.")
    parser.add_argument("path", help="Path to a strategy JSON file or a directory containing *.json files.")
    args = parser.parse_args(argv)

    validator = StrategyDSLValidator()
    try:
        files = iter_strategy_files(Path(args.path))
    except FileNotFoundError:
        print(f"Path not found: {args.path}", file=sys.stderr)
        return 2

    if not files:
        print(f"No strategy JSON files found in {args.path}", file=sys.stderr)
        return 2

    failed = 0
    for file_path in files:
        try:
            validator.validate_file(file_path)
        except DSLValidationError as exc:
            failed += 1
            print(f"[FAIL] {file_path}")
            print(exc)
        else:
            print(f"[PASS] {file_path}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

