"""
dataio.py — raw macro-panel loading, Ludvigson-Ng transform pipeline, targets,
and the train/test split.  ONE place that knows the data; every downstream
script imports from here (the reorg fix from PLAN_rebuild.md §6).

The transform pipeline is a line-for-line port of the authors' own MATLAB
(`macro_data/extracted/.../Makedata.m`, `transx.m`, `standard.m`, `pc_T.m`):

    trimr(macrodat,12,0)                     -> drop first 12 rows (1960:01 start)
    drop columns whose RAW mean is NaN       -> Makedata's isnan(mean) filter
    transx(col, tcode)  (tcode 0 -> 1)       -> per-series stationarity transform
    y = y(49:end,:)                          -> 1964:01 - 2007:12  (528 rows)

Standardisation and PCA live in `pca.py` (full-sample replica for the Stage-0
gate; train-only fit/transform for the leak-free pipeline).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data"          # <repo>/data
RAW_CSV = DATA / "LN2009_macro_panel_raw.csv"
RFS_XLS = DATA / "RFS2009.xls"

H = 12                       # forecast horizon (months)
MATS = (2, 3, 4, 5)         # bond maturities with excess-return targets
TRAIN_END = "1983-12-01"    # last train predictor date
TEST_START = "1984-01-01"   # first test predictor date


# --------------------------------------------------------------------- transx
def transx(x: np.ndarray, tcode: int) -> np.ndarray:
    """Per-series stationarity transform (Ludvigson-Ng `transx.m`).

    1 levels, 2 Δx, 3 Δ²x, 4 ln, 5 Δln, 6 Δ²ln.  For tcodes 4/5/6 a series
    with min < 1e-6 is invalid -> returns an all-NaN column (matches source,
    where the column would later be dropped).  tcode 0 is treated as 1.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    small = 1e-6
    if tcode in (0, 1):
        return x.copy()
    if tcode == 2:
        y = np.empty(n); y[0] = 0.0; y[1:] = x[1:] - x[:-1]; return y
    if tcode == 3:
        y = np.empty(n); y[0] = y[1] = 0.0
        y[2:] = x[2:] - 2 * x[1:-1] + x[:-2]; return y
    if tcode == 4:
        if np.nanmin(x) < small:
            return np.full(n, np.nan)
        return np.log(x)
    if tcode == 5:
        if np.nanmin(x) < small:
            return np.full(n, np.nan)
        lx = np.log(x)
        y = np.empty(n); y[0] = 0.0; y[1:] = lx[1:] - lx[:-1]; return y
    if tcode == 6:
        if np.nanmin(x) < small:
            return np.full(n, np.nan)
        lx = np.log(x)
        y = np.empty(n); y[0] = y[1] = 0.0
        y[2:] = lx[2:] - 2 * lx[1:-1] + lx[:-2]; return y
    return np.full(n, np.nan)


# --------------------------------------------------------------- raw + panel
def load_raw_panel():
    """Return (dates, macrodat 588x131, tcodes[131], names[131]) from the
    exported LN raw panel CSV (row0 = names, row1 = 'Transform:'+tcodes)."""
    raw = pd.read_csv(RAW_CSV, header=0)
    names = list(raw.columns[1:])                      # drop sasdate
    tcodes = raw.iloc[0, 1:].astype(int).to_numpy()
    body = raw.iloc[1:].reset_index(drop=True)
    dates = pd.to_datetime(body.iloc[:, 0], format="%m/%d/%Y")
    macrodat = body.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy()
    return dates.to_numpy(), macrodat, tcodes, names


def build_panel(verbose: bool = False):
    """Transformed macro panel, 1964:01-2007:12 (528 x ~130).

    Returns (dates_528, Y[528, p], kept_names, kept_tcodes, dropped).
    `dropped` lists (name, reason) for the columns removed (raw-mean-NaN or
    transform-invalid), so the Stage-0 report can state exactly what was cut.
    """
    dates, macrodat, tcodes, names = load_raw_panel()
    # trimr(macrodat, 12, 0): drop first 12 rows -> 576 rows, 1960:01 start
    data = macrodat[12:]
    dates_trim = dates[12:]

    kept_cols, kept_names, kept_tcodes, dropped = [], [], [], []
    for j in range(data.shape[1]):
        col = data[:, j]
        if np.isnan(np.mean(col)):               # Makedata: isnan(mean) filter
            dropped.append((names[j], "raw-mean-NaN"))
            continue
        tc = tcodes[j] if tcodes[j] != 0 else 1
        y = transx(col, tc)
        if np.all(np.isnan(y)):
            dropped.append((names[j], f"transform-invalid(tcode{tc})"))
            continue
        kept_cols.append(y)
        kept_names.append(names[j])
        kept_tcodes.append(tc)

    Y = np.column_stack(kept_cols)               # 576 x p
    # y = y(49:end,:) -> drop first 48 -> 1964:01 - 2007:12 (528 rows)
    Y = Y[48:]
    dates_528 = dates_trim[48:]
    if verbose:
        print(f"panel: {Y.shape[0]}x{Y.shape[1]}  dropped {len(dropped)}: "
              f"{[d[0] for d in dropped]}")
    return dates_528, Y, kept_names, np.array(kept_tcodes), dropped


# ------------------------------------------------------------------- targets
def load_targets():
    """Ludvigson-Ng RFS2009.xls: date, CP, published full-sample factors
    f1..f8, and 1-year excess returns yr2..yr5.  480 rows, 1964:01-2003:12."""
    df = pd.read_excel(RFS_XLS, sheet_name="Data Table2", header=5)
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].reset_index(drop=True)
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    keep = ["date", "CP"] + [f"f{i}" for i in range(1, 9)] \
           + [f"yr{a}" for a in MATS]
    return df[keep]


def split_masks(dates):
    """Predictor-date train/test masks (need a valid t+H target -> t <= n-1-H).
    Returns (train_bool, test_bool, tr_idx, te_idx)."""
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    n = len(dates)
    base = np.zeros(n, bool); base[: n - H] = True
    train = base & (dates <= TRAIN_END).to_numpy()
    test = base & (dates >= TEST_START).to_numpy()
    return train, test, np.where(train)[0], np.where(test)[0]


def aligned_panel():
    """The working sample: transformed panel factors joined to RFS targets on
    the overlapping dates 1964:01-2003:12 (480 rows).  Returns a dict with the
    transformed macro panel `Y` restricted to those dates, the target frame,
    and the shared date index — the single input every method sees.
    """
    dpanel, Y, names, tcodes, dropped = build_panel()
    tgt = load_targets()
    dpanel = pd.to_datetime(pd.Series(dpanel))
    pmap = {d: i for i, d in enumerate(dpanel)}
    rows = [pmap[d] for d in tgt["date"] if d in pmap]
    assert len(rows) == len(tgt), (
        f"date misalignment: {len(rows)} of {len(tgt)} target dates found in panel")
    return dict(dates=tgt["date"].to_numpy(), Y=Y[rows], names=names,
                tcodes=tcodes, targets=tgt, dropped=dropped)


if __name__ == "__main__":
    d, Y, names, tcodes, dropped = build_panel(verbose=True)
    print("date range:", pd.Timestamp(d[0]).date(), "->", pd.Timestamp(d[-1]).date())
    a = aligned_panel()
    tr, te, _, _ = split_masks(a["dates"])
    print(f"aligned: Y {a['Y'].shape}  train {tr.sum()} test {te.sum()}")
