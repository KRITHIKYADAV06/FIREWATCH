# src/validation.py
"""
FIREWATCH - validate the MODIS -> VIIRS harmonization on held-out years.

Input : data/processed/cells_monthly.parquet
        (cell_id, cell_lat, cell_lon, year, month, raw_modis, raw_viirs,
         harmonized_count, harmonized_celldays)
        data/processed/ratios.parquet
        (month, ratio_count, ratio_celldays, fit_years, kind)
Output: data/processed/validation.json
        outputs/charts/before_after_harmonization.png
        outputs/charts/monthly_ratios.png
        outputs/charts/predicted_vs_actual.png

Method:
  The monthly MODIS->VIIRS ratio (ratios.parquet, kind == "modis_viirs") is fit on
  2012-2018 (see ratios.fit_years). We check how well it reconstructs the *known*
  VIIRS-scale truth (harmonized_count) on 2019-2025, a period the ratio was never
  fit on:
    truth     = harmonized_count                       (VIIRS / S-NPP scale)
    predicted = raw_modis * ratio_count[month]          (our harmonization)
    naive     = raw_modis                               (no scaling at all)
    pooled    = raw_modis * one pooled ratio            (one number, no seasonality)
  Metrics are computed twice: once on India-wide monthly totals, and once per
  cell-month (restricted to cell-months where truth or predicted is > 0, so empty
  cells do not dilute the error with trivial 0-vs-0 matches).

Run from the repo root:
    python src/validation.py
    python src/validation.py --cells data/processed/cells_monthly.parquet \
        --ratios data/processed/ratios.parquet --out data/processed
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TEST_YEARS = list(range(2019, 2026))          # 2019-2025 inclusive
EXPECTED_FIT_YEARS = "2012-2018"              # what ratios.fit_years should start with


# --------------------------------------------------------------------- loading
def load_inputs(cells_path, ratios_path):
    """Read the two required parquet files. Raises FileNotFoundError with a
    clear message if either is missing (we never invent the numbers inside)."""
    cells_path, ratios_path = Path(cells_path), Path(ratios_path)
    missing = [p for p in (cells_path, ratios_path) if not p.exists()]
    if missing:
        names = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"validation needs real harmonized data, but these files are missing: {names}. "
            "Run src/ingest.py then src/harmonize.py first (needs a FIRMS_MAP_KEY in .env)."
        )
    cells = pd.read_parquet(cells_path)
    ratios = pd.read_parquet(ratios_path)
    return cells, ratios


def check_fit_years(ratios):
    """Confirm the MODIS->VIIRS ratio was fit on 2012-2018, as the method requires."""
    mv = ratios[ratios["kind"] == "modis_viirs"]
    if mv.empty:
        raise ValueError("ratios.parquet has no rows with kind == 'modis_viirs'")
    fit_years = mv["fit_years"].iloc[0]
    if not str(fit_years).startswith(EXPECTED_FIT_YEARS):
        raise ValueError(
            f"ratios.parquet says fit_years={fit_years!r}, expected it to start with "
            f"{EXPECTED_FIT_YEARS!r}. The fit window (and this validation's 2019-2025 "
            "holdout) would no longer be independent."
        )
    return fit_years


# -------------------------------------------------------------------- pooling
def pooled_ratio(cells, fit_start=2012, fit_end=2018):
    """One India-wide ratio (no month-to-month seasonality), fit on the same
    2012-2018 window as the monthly ratios: sum(truth) / sum(raw_modis)."""
    fit = cells[cells["year"].between(fit_start, fit_end)]
    modis_total = fit["raw_modis"].sum(min_count=1)
    truth_total = fit["harmonized_count"].sum(min_count=1)
    if not modis_total or modis_total <= 0:
        raise ValueError("no raw_modis data in the 2012-2018 fit window to pool a ratio from")
    return float(truth_total / modis_total)


# ------------------------------------------------------------------- metrics
def mae(actual, pred):
    return float(np.mean(np.abs(actual - pred)))


def mape(actual, pred):
    """Mean absolute percentage error, over rows where actual > 0 only
    (a percentage error against a true 0 is undefined)."""
    mask = actual > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100.0)


def pearson_r(actual, pred):
    if len(actual) < 2 or np.std(actual) == 0 or np.std(pred) == 0:
        return float("nan")
    return float(np.corrcoef(actual, pred)[0, 1])


def _metric_block(actual, pred, naive, pooled_pred):
    return {
        "mae": mae(actual, pred),
        "mape": mape(actual, pred),
        "r": pearson_r(actual, pred),
        "naive_mae": mae(actual, naive),
        "naive_mape": mape(actual, naive),
        "pooled_mae": mae(actual, pooled_pred),
        "pooled_mape": mape(actual, pooled_pred),
        "pooled_r": pearson_r(actual, pooled_pred),
        "n": int(len(actual)),
    }


# --------------------------------------------------------------- the holdout
def build_holdout(cells, ratios, pooled_r):
    """Attach predicted/naive/pooled columns to the 2019-2025 cell-months."""
    mv = ratios.loc[ratios["kind"] == "modis_viirs", ["month", "ratio_count"]]
    test = cells[cells["year"].isin(TEST_YEARS)].merge(mv, on="month", how="left")
    if test["ratio_count"].isna().any():
        raise ValueError("some test months have no modis_viirs ratio in ratios.parquet")

    test = test.copy()
    test["actual"] = test["harmonized_count"].fillna(0.0)
    test["raw_modis"] = test["raw_modis"].fillna(0.0)
    test["predicted"] = test["raw_modis"] * test["ratio_count"]
    test["naive"] = test["raw_modis"]
    test["pooled_pred"] = test["raw_modis"] * pooled_r
    return test


def india_monthly_metrics(test):
    """Sum every cell to one India-wide number per (year, month), then score."""
    by_ym = (test.groupby(["year", "month"])[["actual", "predicted", "naive", "pooled_pred"]]
                 .sum()
                 .reset_index())
    return _metric_block(by_ym["actual"].to_numpy(), by_ym["predicted"].to_numpy(),
                          by_ym["naive"].to_numpy(), by_ym["pooled_pred"].to_numpy())


def cell_month_metrics(test):
    """Score every cell-month where the truth or our prediction is > 0."""
    keep = (test["actual"] > 0) | (test["predicted"] > 0)
    sub = test[keep]
    return _metric_block(sub["actual"].to_numpy(), sub["predicted"].to_numpy(),
                          sub["naive"].to_numpy(), sub["pooled_pred"].to_numpy())


# ----------------------------------------------------------------------- run
def run_validation(cells_path, ratios_path):
    """Load inputs from disk, then score them (see run_validation_inmemory)."""
    cells, ratios = load_inputs(cells_path, ratios_path)
    return run_validation_inmemory(cells, ratios)


def run_validation_inmemory(cells, ratios):
    """Compute every metric from already-loaded DataFrames. Split out from
    run_validation() so tests can pass in synthetic data without touching disk."""
    fit_years = check_fit_years(ratios)
    pooled_r = pooled_ratio(cells)
    test = build_holdout(cells, ratios, pooled_r)

    india = india_monthly_metrics(test)
    cell = cell_month_metrics(test)

    result = {
        "mae": india["mae"],
        "mape": india["mape"],
        "r": india["r"],
        "naive_mae": india["naive_mae"],
        "naive_mape": india["naive_mape"],
        "test_years": TEST_YEARS,
        "fit_years": fit_years,
        "pooled_ratio": pooled_r,
        "pooled_mae": india["pooled_mae"],
        "pooled_mape": india["pooled_mape"],
        "pooled_r": india["pooled_r"],
        "cell_mae": cell["mae"],
        "cell_mape": cell["mape"],
        "cell_r": cell["r"],
        "cell_naive_mae": cell["naive_mae"],
        "cell_naive_mape": cell["naive_mape"],
        "cell_pooled_mae": cell["pooled_mae"],
        "cell_pooled_mape": cell["pooled_mape"],
        "cell_pooled_r": cell["pooled_r"],
        "cell_n": cell["n"],
    }
    return result, test


# --------------------------------------------------------------------- charts
def save_charts(cells, ratios, test, out_dir):
    """Save the 3 required PNGs. Needs matplotlib; raises ImportError if absent
    so the caller can skip this step and say so, instead of silently doing nothing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Before/after harmonization, India monthly totals, 2003-2026
    yearly = (cells.assign(ym=pd.to_datetime(dict(year=cells["year"], month=cells["month"], day=1)))
                   .groupby("ym")[["raw_modis", "raw_viirs", "harmonized_count"]]
                   .sum(min_count=1)
                   .reset_index()
                   .sort_values("ym"))
    naive_splice = yearly["raw_modis"].fillna(0) + yearly["raw_viirs"].fillna(0)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(yearly["ym"], naive_splice, label="before (MODIS + VIIRS, unscaled)", color="#d55e00")
    ax.plot(yearly["ym"], yearly["harmonized_count"], label="after (harmonized, VIIRS scale)", color="#0072b2")
    ax.set_title("India-wide monthly hotspot count: before vs after harmonization")
    ax.set_xlabel("month")
    ax.set_ylabel("hotspot count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "before_after_harmonization.png", dpi=150)
    plt.close(fig)

    # 2. Monthly MODIS->VIIRS ratios, with the S-NPP (Feb 2012) and NOAA-20
    #    (Apr 2018) switches marked
    mv = ratios[ratios["kind"] == "modis_viirs"].sort_values("month")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(mv["month"], mv["ratio_count"], marker="o", color="#0072b2")
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("calendar month")
    ax.set_ylabel("ratio_count (VIIRS / MODIS)")
    ax.set_title("Monthly MODIS -> VIIRS ratio (fit 2012-2018)")
    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")
    ax.annotate("S-NPP starts Feb 2012", xy=(0.02, 0.95), xycoords="axes fraction",
                fontsize=8, color="#444444")
    ax.annotate("NOAA-20 overlap starts Apr 2018", xy=(0.02, 0.89), xycoords="axes fraction",
                fontsize=8, color="#444444")
    fig.tight_layout()
    fig.savefig(out_dir / "monthly_ratios.png", dpi=150)
    plt.close(fig)

    # 3. Predicted vs actual, 2019-2025 (India-monthly, so the plot is readable)
    by_ym = test.groupby(["year", "month"])[["actual", "predicted"]].sum().reset_index()
    lim = float(max(by_ym["actual"].max(), by_ym["predicted"].max()) * 1.05) if len(by_ym) else 1.0
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(by_ym["actual"], by_ym["predicted"], color="#0072b2", alpha=0.8)
    ax.plot([0, lim], [0, lim], color="grey", linestyle="--", linewidth=1, label="y = x")
    ax.set_xlabel("actual (harmonized_count)")
    ax.set_ylabel("predicted (raw_modis x ratio_count)")
    ax.set_title("Predicted vs actual, India-monthly, 2019-2025")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "predicted_vs_actual.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------- CLI
def main():
    p = argparse.ArgumentParser(description="FIREWATCH validation (2019-2025 holdout)")
    p.add_argument("--cells", default="data/processed/cells_monthly.parquet")
    p.add_argument("--ratios", default="data/processed/ratios.parquet")
    p.add_argument("--out", default="data/processed")
    p.add_argument("--charts", default="outputs/charts")
    args = p.parse_args()

    result, test = run_validation(args.cells, args.ratios)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "validation.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {out_dir / 'validation.json'}")
    print(json.dumps(result, indent=2))

    cells, ratios = load_inputs(args.cells, args.ratios)
    try:
        save_charts(cells, ratios, test, args.charts)
        print(f"Saved 3 charts to {args.charts}/")
    except ImportError:
        print("matplotlib is not installed - skipped the 3 PNG charts.")


if __name__ == "__main__":
    main()
