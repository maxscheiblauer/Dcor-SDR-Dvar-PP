"""ch04_shengyin_2016_design.py — the Sheng & Yin (2016) simulation design

Their published design, run with both solvers: the gradient search of
this work and the sequential-quadratic-programming solver of the paper.
Includes the p in {6, 20, 50, 100} scaling block.

Thesis: Chapter 4, tab:p1-fourway (lower panel) and the fig:p1-cost data
Writes: results/results_sheng_yin_2016.csv
Original: PP_Dcor/sheng_yin_2016_study.py on the thesis branch.

Runtime: about 60 minutes (see README for the machine).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from csvout import pin_blas_threads                      # noqa: E402

pin_blas_threads()   # must precede numpy: see csvout.pin_blas_threads

if hasattr(sys.stdout, "reconfigure"):      # tables below contain non-ASCII
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from designs.sheng_yin_2016 import main                                # noqa: E402

if __name__ == "__main__":
    main()
