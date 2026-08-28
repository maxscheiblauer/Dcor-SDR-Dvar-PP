"""ch05_shape_table.py — how the distance-variance index orders distributions

Distance variance and excess kurtosis for eleven laws standardised to
unit variance, plus the separation sweep for the bimodal mixture.  This
is the evidence that the index measures dispersion and not
non-Gaussianity: the Gaussian sits in the middle of the ordering.

Thesis: Chapters 1 and 2, tab:bg-shape
Writes: results/results_shape.csv
Original: Dvar-PP/make_shape_table.py on the thesis branch.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from csvout import pin_blas_threads                      # noqa: E402

pin_blas_threads()   # must precede numpy: see csvout.pin_blas_threads

if hasattr(sys.stdout, "reconfigure"):      # tables below contain non-ASCII
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from designs.shapes import main                                # noqa: E402

if __name__ == "__main__":
    main()
