"""Utility for bundling multiple CSV files in a directory into a single Excel workbook.

Features:
  * Each CSV becomes an individual worksheet.
  * Worksheet names derived from (and limited by) the original file names.
  * Handles name collisions & invalid Excel characters gracefully.
  * Optional glob pattern filter and deterministic ordering.

Example (module execution):
	python -m src.util.csv_bundler data/pattern pattern_bundle.xlsx
	# or (output optional)
	python -m src.util.csv_bundler data/pattern

The function `bundle_csvs_to_excel` can also be imported and reused.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional


INVALID_SHEET_CHARS = re.compile(r"[:\\/?*\[\]]")  # Characters Excel forbids in sheet names
MAX_SHEET_NAME_LEN = 31  # Excel sheet name length limit


def _sanitize_sheet_name(name: str) -> str:
	"""Return a sanitized Excel sheet name within Excel's constraints.

	- Removes invalid characters.
	- Truncates to 31 chars.
	- Strips leading/trailing whitespace.
	"""
	cleaned = INVALID_SHEET_CHARS.sub("_", name).strip()
	if not cleaned:
		cleaned = "sheet"
	if len(cleaned) > MAX_SHEET_NAME_LEN:
		cleaned = cleaned[:MAX_SHEET_NAME_LEN]
	return cleaned


def _dedupe(names: Iterable[str]) -> List[str]:
	"""Ensure sheet names are unique by appending a numeric suffix when needed."""
	seen = {}
	result = []
	for name in names:
		base = name
		if base not in seen:
			seen[base] = 0
			result.append(base)
			continue
		# Need to find a unique suffix
		idx = seen[base] + 1
		while True:
			candidate = base
			# Reserve up to 4 chars for suffix like _12 (ensure length limit)
			suffix = f"_{idx}"
			if len(candidate) + len(suffix) > MAX_SHEET_NAME_LEN:
				candidate = candidate[: MAX_SHEET_NAME_LEN - len(suffix)]
			candidate = f"{candidate}{suffix}"
			if candidate not in seen:
				seen[base] = idx
				seen[candidate] = 0
				result.append(candidate)
				break
			idx += 1
	return result


def bundle_csvs_to_excel(
	directory: str | os.PathLike,
	output_path: Optional[str | os.PathLike] = None,
) -> Path:
	"""Bundle all CSV files in a directory into a single Excel file.

	Parameters
	----------
	directory : str | Path
		Directory containing CSV files.
	output_path : str | Path | None
		Output Excel file path (.xlsx). If None, uses '<directory_name>.xlsx' in the
		same parent directory.

	Returns
	-------
	Path
		The path to the created Excel workbook.

	Raises
	------
	FileNotFoundError
		If directory does not exist or no CSV files match.
	RuntimeError
		If pandas is not installed.
	"""
	try:
		import pandas as pd  # type: ignore
	except Exception as exc:  # pragma: no cover - import guard
		raise RuntimeError(
			"pandas is required for bundle_csvs_to_excel. Please install with 'pip install pandas openpyxl'."
		) from exc

	dir_path = Path(directory).expanduser().resolve()
	if not dir_path.is_dir():
		raise FileNotFoundError(f"Directory not found: {dir_path}")

	pattern_path = str(dir_path / "*.csv")
	csv_files = [Path(p) for p in glob.glob(pattern_path) if Path(p).is_file()]
	if not csv_files:
		raise FileNotFoundError(f"No CSV files found in {dir_path}.")

	# Deterministic ordering
	csv_files.sort(key=lambda p: p.name.lower())

	raw_sheet_names = [_sanitize_sheet_name(p.stem) for p in csv_files]
	sheet_names = _dedupe(raw_sheet_names)

	if output_path is None:
		output_path = dir_path.parent / f"{dir_path.name}.xlsx"
	output_path = Path(output_path).expanduser().resolve()
	output_path.parent.mkdir(parents=True, exist_ok=True)

	# Write workbook
	with pd.ExcelWriter(output_path) as writer:  # type: ignore[arg-type]
		for path_obj, sheet in zip(csv_files, sheet_names):
			df = pd.read_csv(path_obj, encoding="utf-8")
			df.to_excel(writer, sheet_name=sheet, index=False)

	return output_path


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bundle all CSV files in a directory into a single Excel workbook (each CSV -> one sheet)."
    )
    parser.add_argument("directory", help="Directory containing CSV files")
    parser.add_argument(
        "output",
        nargs="?",
        help="Output Excel file path (.xlsx). If omitted uses <directory_name>.xlsx in the parent directory.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:  # pragma: no cover - CLI wrapper
    args = _parse_args(argv)
    output = bundle_csvs_to_excel(directory=args.directory, output_path=args.output)
    print(f"Created Excel workbook: {output}")


if __name__ == "__main__":  # pragma: no cover
    main()
