"""ch06_outlier_recovery.py — full-sample outlier recovery

Which criteria isolate the two 2001 months when every method is fitted on
the whole sample: distance variance plain and biloop, PCA, and the
minimum-covariance-determinant Mahalanobis distance.

Thesis: Chapter 6, the outlier case study
Writes: results/outlier_recovery.csv, results/outlier_recovery_agg.csv
Original: Real Data Experiment/src/outlier_recovery.py on the thesis branch.

Runtime: about 13 minutes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from csvout import pin_blas_threads                      # noqa: E402

pin_blas_threads()   # must precede numpy: see csvout.pin_blas_threads

if hasattr(sys.stdout, "reconfigure"):      # tables below contain non-ASCII
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from realdata.recovery import main                                # noqa: E402

if __name__ == "__main__":
    main()
