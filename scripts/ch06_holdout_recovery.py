"""ch06_holdout_recovery.py — the same comparison with the pair held out

Design A: the two 2001 months are removed from every fit, so a criterion
cannot have been fitted to the observations it is then asked to flag.

Thesis: Chapter 6, the outlier case study
Writes: results/holdout_recovery.csv, results/holdout_recovery_agg.csv
Original: Real Data Experiment/src/holdout_recovery.py on the thesis branch.

Runtime: about 19 minutes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from csvout import pin_blas_threads                      # noqa: E402

pin_blas_threads()   # must precede numpy: see csvout.pin_blas_threads

if hasattr(sys.stdout, "reconfigure"):      # tables below contain non-ASCII
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from realdata.holdout import main                                # noqa: E402

if __name__ == "__main__":
    main()
