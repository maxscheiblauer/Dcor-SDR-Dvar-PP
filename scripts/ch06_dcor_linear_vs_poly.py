"""ch06_dcor_linear_vs_poly.py — Chapter 6 experiments 1 and 2

Experiment 1: whether the polynomial second stage adds anything once the
first stage is chosen with the response.  Experiment 2: the Diebold-Mariano
tests for the leaked dCor-linear specification against its anchors.

Thesis: Chapter 6, experiments 1 and 2
Writes: results/exp1_dcor_poly_vs_lin.csv,
        results/exp2_leaked_dcorlin_dm.csv
Original: Real Data Experiment/src/dcorlin_experiments.py on the thesis branch.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from csvout import pin_blas_threads                      # noqa: E402

pin_blas_threads()   # must precede numpy: see csvout.pin_blas_threads

if hasattr(sys.stdout, "reconfigure"):      # tables below contain non-ASCII
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from realdata.dcorlin import main                                # noqa: E402

if __name__ == "__main__":
    main()
