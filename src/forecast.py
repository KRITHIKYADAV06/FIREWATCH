# src/forecast.py
"""
FIREWATCH - experimental next-month fire-activity forecast, per 0.1-degree cell.

Input : data/processed/cells_monthly.parquet
        (cell_id, cell_lat, cell_lon, year, month, raw_modis, raw_viirs,
         harmonized_count, harmonized_celldays)
Output: data/processed/forecast.parquet
        (cell_id, cell_lat, cell_lon, year, month, predicted, label)
        data/processed/forecast_metrics.json
        outputs/charts/forecast_backtest.png

This is NOT a fire-risk or burned-area model. It predicts next month's
harmonized hotspot COUNT per cell from its own recent history - nothing more.
Every forecast row is labelled "experimental".

Method:
  1. Build a full cell x month grid from 2013-01 to the latest month in the
     data (missing cell-months = 0 hotspots: the "zero rule"). Only cells with
     >= 5 fires over the whole period are kept (too little signal otherwise).
  2. Features, all built from the PAST only (no leakage):
       lag1, lag2, lag3, lag12            - harmonized_count N months back
       mean_last3                         - mean of lag1..lag3
       month_median_prevyears             - median of this calendar month in
                                             all STRICTLY EARLIER years
       month_sin, month_cos               - calendar-month seasonality
       cell_lat, cell_lon                 - location
  3. Model: HistGradientBoostingRegressor(loss="poisson") (counts are >= 0 and
     skewed, so a Poisson loss fits better than squared error).
     Train <= 2023, validate 2024, test 2025 (time-ordered: never train on the
     future of a split used to judge it).
  4. Baselines: seasonal naive (= lag12) and climatology (= month_median_prevyears).
     We report 2025 MAE / RMSE / top-10% hit rate for the model AND both
     baselines, and say plainly if the model does not beat them.

Run from the repo root:
    python src/forecast.py
    python src/forecast.py --cells data/processed/cells_monthly.parquet --out data/processed
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

GRID_START_YM = 201301          # 2013-01, as the spec asks
MIN_TOTAL_FIRES = 5             # drop cells with fewer than this many fires in total
TRAIN_END_YEAR = 2023
VALID_YEAR = 2024
TEST_YEAR = 2025
LAGS = (1, 2, 3, 12)

FEATURE_COLS = ["lag1", "lag2", "lag3", "lag12", "mean_last3",
                 "month_median_prevyears", "month_sin", "month_cos",
                 "cell_lat", "cell_lon"]
FORECAST_COLS = ["cell_id", "cell_lat", "cell_lon", "year", "month", "predicted", "label"]


# --------------------------------------------------------------------- loading
def load_cells_monthly(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"forecast needs real harmonized data, but {path} is missing. "
            "Run src/ingest.py then src/harmonize.py first (needs a FIRMS_MAP_KEY in .env)."
        )
    return pd.read_parquet(path, columns=["cell_id", "cell_lat", "cell_lon", "year",
                                           "month", "harmonized_count"])


# ------------------------------------------------------------------- the grid
def build_full_grid(cells_monthly, grid_start_ym=GRID_START_YM, min_total_fires=MIN_TOTAL_FIRES):
    """Full cell x month grid, zero-filled, cells with too little signal dropped."""
    cm = cells_monthly.copy()
    cm["ym"] = cm["year"].astype(int) * 100 + cm["month"].astype(int)

    totals = cm.groupby("cell_id")["harmonized_count"].sum(min_count=1)
    keep_cells = totals[totals.fillna(0) >= min_total_fires].index
    cm = cm[cm["cell_id"].isin(keep_cells)]
    if cm.empty:
        raise ValueError(f"no cell has >= {min_total_fires} fires in total - nothing to forecast")

    coords = cm.drop_duplicates("cell_id").set_index("cell_id")[["cell_lat", "cell_lon"]]

    last_ym = int(cm["ym"].max())
    months = pd.period_range(f"{grid_start_ym // 100}-{grid_start_ym % 100:02d}",
                              f"{last_ym // 100}-{last_ym % 100:02d}", freq="M")
    grid = pd.MultiIndex.from_product([keep_cells, months], names=["cell_id", "period"]).to_frame(index=False)
    grid["year"] = grid["period"].dt.year.astype(int)
    grid["month"] = grid["period"].dt.month.astype(int)
    grid = grid.drop(columns="period")

    cm_small = cm[["cell_id", "year", "month", "harmonized_count"]]
    full = grid.merge(cm_small, on=["cell_id", "year", "month"], how="left")
    full["harmonized_count"] = full["harmonized_count"].fillna(0.0)   # zero rule
    full = full.merge(coords, on="cell_id", how="left")
    return full.sort_values(["cell_id", "year", "month"]).reset_index(drop=True)


# ---------------------------------------------------------------- feature eng
def add_features(full):
    """Add lag / rolling / seasonal / location features. Every feature for a
    row at (cell, year, month) only uses STRICTLY EARLIER months of that cell,
    so nothing here can leak the label into its own features."""
    df = full.sort_values(["cell_id", "year", "month"]).reset_index(drop=True)
    g = df.groupby("cell_id")["harmonized_count"]

    for lag in LAGS:
        df[f"lag{lag}"] = g.shift(lag)

    df["mean_last3"] = df[["lag1", "lag2", "lag3"]].mean(axis=1)

    # same-month median of all STRICTLY EARLIER years, per cell
    df["month_median_prevyears"] = np.nan
    for (cell_id, month), idx in df.groupby(["cell_id", "month"]).groups.items():
        sub = df.loc[idx].sort_values("year")
        past_median = sub["harmonized_count"].expanding().median().shift(1)
        df.loc[sub.index, "month_median_prevyears"] = past_median.values

    angle = 2 * np.pi * df["month"] / 12.0
    df["month_sin"] = np.sin(angle)
    df["month_cos"] = np.cos(angle)

    df["label"] = df["harmonized_count"]            # what we are predicting (NEXT month, via the row shift below)
    return df


def make_supervised(full):
    """Turn the per-cell-month table into (features at month m) -> (count at month m+1)."""
    df = add_features(full)
    df = df.sort_values(["cell_id", "year", "month"]).reset_index(drop=True)
    g = df.groupby("cell_id")
    df["target"] = g["harmonized_count"].shift(-1)       # next month's count (the label)
    df["target_year"] = g["year"].shift(-1)
    df["target_month"] = g["month"].shift(-1)
    df["seasonal_naive"] = df["lag12"]                   # same month last year, known at prediction time
    df = df.dropna(subset=FEATURE_COLS + ["target", "target_year", "target_month"]).copy()
    df["target_year"] = df["target_year"].astype(int)
    df["target_month"] = df["target_month"].astype(int)
    return df


# ----------------------------------------------------------------- the model
def _hgb_regressor():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(loss="poisson", random_state=0)


def split_train_valid_test(sup):
    train = sup[sup["target_year"] <= TRAIN_END_YEAR]
    valid = sup[sup["target_year"] == VALID_YEAR]
    test = sup[sup["target_year"] == TEST_YEAR]
    return train, valid, test


def fit_model(train):
    model = _hgb_regressor()
    model.fit(train[FEATURE_COLS], train["target"])
    return model


# ------------------------------------------------------------------- metrics
def mae(actual, pred):
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(pred))))


def rmse(actual, pred):
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(pred)) ** 2)))


def top_decile_hit_rate(actual, pred):
    """Of the cell-months in the TRUE top 10% busiest, what fraction did the
    model also rank in ITS predicted top 10%?"""
    actual, pred = np.asarray(actual), np.asarray(pred)
    n = len(actual)
    if n == 0:
        return float("nan")
    k = max(1, int(np.ceil(n * 0.10)))
    true_top = set(np.argsort(-actual)[:k])
    pred_top = set(np.argsort(-pred)[:k])
    return float(len(true_top & pred_top) / len(true_top))


def evaluate(actual, pred):
    return {"mae": mae(actual, pred), "rmse": rmse(actual, pred),
            "top10pct_hit_rate": top_decile_hit_rate(actual, pred)}


# ----------------------------------------------------------------------- run
def run_forecast(cells_path):
    cells_monthly = load_cells_monthly(cells_path)
    full = build_full_grid(cells_monthly)
    sup = make_supervised(full)
    train, valid, test = split_train_valid_test(sup)
    if train.empty or test.empty:
        raise ValueError("not enough history to form a train/test split - check the data range")

    model = fit_model(train)

    metrics = {}
    for name, split in (("train", train), ("valid", valid), ("test", test)):
        if split.empty:
            continue
        pred = model.predict(split[FEATURE_COLS])
        clim = split["month_median_prevyears"]
        seas_naive = split["seasonal_naive"]
        metrics[name] = {
            "model": evaluate(split["target"], pred),
            "seasonal_naive": evaluate(split["target"], seas_naive),
            "climatology": evaluate(split["target"], clim),
            "n": int(len(split)),
        }
    test_metrics = metrics.get("test", {})
    if test_metrics:
        beats_seasonal = test_metrics["model"]["mae"] < test_metrics["seasonal_naive"]["mae"]
        beats_climatology = test_metrics["model"]["mae"] < test_metrics["climatology"]["mae"]
        metrics["beats_seasonal_naive"] = bool(beats_seasonal)
        metrics["beats_climatology"] = bool(beats_climatology)
        metrics["honest_summary"] = (
            "model beats both baselines on 2025 MAE" if beats_seasonal and beats_climatology
            else "model does NOT beat both baselines on 2025 MAE - use with caution"
        )
    metrics["train_end_year"] = TRAIN_END_YEAR
    metrics["valid_year"] = VALID_YEAR
    metrics["test_year"] = TEST_YEAR

    # Next-month forecast: one row per kept cell, using the LAST available month
    # of real data as the "current" month, predicting the month after it.
    last = (full.sort_values(["cell_id", "year", "month"])
                .groupby("cell_id").tail(1))
    next_full = add_features(full)
    next_features = (next_full.sort_values(["cell_id", "year", "month"])
                               .groupby("cell_id").tail(1))
    next_features = next_features.dropna(subset=FEATURE_COLS)
    next_ym = pd.Period(f"{int(last['year'].iloc[0])}-{int(last['month'].iloc[0]):02d}", freq="M") + 1

    forecast = next_features[["cell_id", "cell_lat", "cell_lon"]].copy()
    forecast["predicted"] = model.predict(next_features[FEATURE_COLS])
    forecast["predicted"] = forecast["predicted"].clip(lower=0.0)
    forecast["year"] = next_ym.year
    forecast["month"] = next_ym.month
    forecast["label"] = "experimental"
    forecast = forecast[FORECAST_COLS].reset_index(drop=True)

    return model, metrics, forecast, test


# --------------------------------------------------------------------- chart
def save_backtest_chart(test, model, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred = model.predict(test[FEATURE_COLS])
    by_ym = (test.assign(predicted=pred)
                 .groupby(["target_year", "target_month"])[["target", "predicted", "seasonal_naive"]]
                 .sum()
                 .reset_index())
    by_ym["ym"] = pd.to_datetime(dict(year=by_ym["target_year"], month=by_ym["target_month"], day=1))
    by_ym = by_ym.sort_values("ym")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(by_ym["ym"], by_ym["target"], label="actual", color="#0072b2", marker="o")
    ax.plot(by_ym["ym"], by_ym["predicted"], label="model", color="#d55e00", marker="o")
    ax.plot(by_ym["ym"], by_ym["seasonal_naive"], label="seasonal naive", color="grey", linestyle="--")
    ax.set_title(f"Next-month forecast backtest, {TEST_YEAR} (India total, experimental)")
    ax.set_ylabel("harmonized hotspot count")
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------- CLI
def main():
    p = argparse.ArgumentParser(description="FIREWATCH next-month forecast (experimental)")
    p.add_argument("--cells", default="data/processed/cells_monthly.parquet")
    p.add_argument("--out", default="data/processed")
    p.add_argument("--charts", default="outputs/charts")
    args = p.parse_args()

    model, metrics, forecast, test = run_forecast(args.cells)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    forecast.to_parquet(out_dir / "forecast.parquet", index=False)
    with open(out_dir / "forecast_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved {out_dir / 'forecast.parquet'} ({len(forecast)} cells) and "
          f"{out_dir / 'forecast_metrics.json'}")
    print(json.dumps(metrics, indent=2))

    try:
        save_backtest_chart(test, model, Path(args.charts) / "forecast_backtest.png")
        print(f"Saved {args.charts}/forecast_backtest.png")
    except ImportError:
        print("matplotlib is not installed - skipped forecast_backtest.png")


if __name__ == "__main__":
    main()
