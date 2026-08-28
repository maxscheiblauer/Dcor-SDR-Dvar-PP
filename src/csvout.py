"""csvout.py — the one CSV writer every script in ``scripts/`` uses.

Two header lines, then the data:

    # script: ch03_gradient_check.py
    # seeds: 42, 7, 123, 2024, 5, 2026, 17, 99, 777, 31337
    index,seed,n,p,best_h,err_at_best_h,err_at_1e5
    ...

The seed line is the record that matters: every number in this repository is a
function of the code, the seed list and the library versions, and the first two are
therefore stated in the file itself.  ``pandas.read_csv(path, comment="#")`` reads
these files unchanged.

Byte-for-byte compatibility is deliberate.  The formatting below — ``csv.DictWriter``
with ``lineterminator="\\n"``, column order from ``fieldnames`` or from the first
row's keys, ``to_csv(index=False)`` for a frame, UTF-8, LF newlines — reproduces what
wrote the CSVs shipped in ``results/``, so a rerun can be compared against them with
a plain diff of everything below the header.
"""
from __future__ import annotations

import csv
import sys
from io import StringIO
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

#: Where every script writes. Resolved from this file, so the working directory does
#: not matter: ``src/csvout.py`` -> ``<repo>/results``.
RESULTS = Path(__file__).resolve().parents[1] / "results"

__all__ = ["RESULTS", "write_csv", "read_csv", "pin_blas_threads"]


def pin_blas_threads() -> None:
    """Restrict every BLAS backend to one thread.  Call before importing numpy.

    Not a performance setting.  The thread count changes the order of the
    floating-point reductions inside the distance-matrix products, which changes the
    result in the last bits, which decides the ``s_xy <= 0`` comparison that selects
    the degenerate-point branch of the dCor gradient.  The published numbers were
    produced single-threaded, and a multi-threaded rerun does not reproduce them
    exactly.  Setting these variables after numpy is imported has no effect, which is
    why every script calls this first.
    """
    import os

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"


def write_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]] | Any,
    seeds: Sequence[int] | int | None = None,
    fieldnames: Sequence[str] | None = None,
    script: str | None = None,
    float_format: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Write ``rows`` to ``path`` under the header.  Returns the header as a dict.

    ``rows`` is either an iterable of mappings or an object with a ``to_csv`` method
    (a pandas frame).  Column order comes from ``fieldnames`` if given, otherwise from
    the first row.  A relative ``path`` is taken relative to ``results/``.

    Anything passed as ``**extra`` becomes a further ``# key: value`` line — the
    budgets and replicate counts a script was run with, which are part of what the
    numbers depend on.  Sequences are joined with commas.
    """
    path = Path(path)
    if not path.is_absolute():
        path = RESULTS / path

    if seeds is None:
        seed_list: list[int] = []
    elif isinstance(seeds, int):
        seed_list = [seeds]
    else:
        seed_list = list(seeds)

    name = script or (Path(sys.argv[0]).name if sys.argv and sys.argv[0]
                      else "interactive")
    record: dict[str, Any] = {"script": name, "seeds": seed_list}
    record.update(extra)

    def _render(value: Any) -> str:
        if isinstance(value, (list, tuple)):
            return ", ".join(str(v) for v in value)
        return str(value)

    header = "".join(f"# {key}: {_render(value)}\n"
                     for key, value in record.items())

    if hasattr(rows, "to_csv"):  # pandas frame or series
        body = rows.to_csv(index=False, float_format=float_format,
                           lineterminator="\n")
    else:
        rows = list(rows)
        if not rows:
            raise ValueError(f"refusing to write {path.name} with no rows")
        names = list(fieldnames) if fieldnames else list(rows[0].keys())
        buf = StringIO()
        writer = csv.DictWriter(buf, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if float_format is not None:
                row = {k: (float_format % v if isinstance(v, float) else v)
                       for k, v in row.items()}
            writer.writerow(row)
        body = buf.getvalue()

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + body, encoding="utf-8", newline="\n")
    return record


def read_csv(path: str | Path, **kwargs: Any):
    """``pandas.read_csv`` with the header comments skipped."""
    import pandas as pd

    path = Path(path)
    if not path.is_absolute():
        path = RESULTS / path
    kwargs.setdefault("comment", "#")
    return pd.read_csv(path, **kwargs)
