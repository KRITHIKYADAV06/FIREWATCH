# src/harmonize.py
"""
FIREWATCH - harmonize MODIS Aqua and VIIRS onto ONE scale (VIIRS S-NPP).

Input : data/processed/cell_days.parquet
        (cell_id, cell_lat, cell_lon, date, sensor, satellite, n_hotspots, frp_sum)
Output: data/processed/ratios.parquet
        (month, ratio_count, ratio_celldays, fit_years, kind)
        data/processed/cells_monthly.parquet
        (cell_id, cell_lat, cell_lon, year, month, raw_modis, raw_viirs,
         harmonized_count, harmonized_celldays)

Method (Krithik, v3 spec):
  1. Only Aqua, S-NPP, NOAA-20 are used. NOAA-21 is ignored here.
  2. Per cell-month: count = sum of hotspots, celldays = number of active days.
  3. MODIS->VIIRS ratio per calendar month m, pooled India-wide, fit 2012-2018:
        r_m = sum(S-NPP) / sum(Aqua)   (Jan 2012 skipped: S-NPP starts 20 Jan)
     If a month's Aqua total is tiny, use the pooled annual ratio and flag it.
  4. NOAA-20->S-NPP ratio per month on the overlap (Apr 2018 -> last S-NPP month).
  5. Harmonized series (S-NPP scale):
        before Feb 2012            -> Aqua x r_m
        Feb 2012 .. switch month-1 -> S-NPP as observed
        switch month onwards       -> NOAA-20 / n20_ratio_m
     The switch month is the month after S-NPP data ends (Aug 2026 with archive
     only; Nov 2026 once NRT up to 2 Nov is added). Each month uses ONE VIIRS
     satellite, so S-NPP and NOAA-20 are never added together.
  6. raw_modis / raw_viirs stay RAW (unscaled). NaN = sensor had no data then.
     A cell-month missing from the file means 0 detections from every sensor.

Run from the repo root:
    python src/harmonize.py
    python src/harmonize.py --cells data/processed/cell_days.parquet --out data/processed
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CORE_SATS = ["Aqua", "S-NPP", "NOAA-20"]
FIT_START, FIT_END = 2012, 2018
SNPP_FIRST_FULL = 201202     # yyyymm: first full S-NPP month (data starts 20 Jan 2012)
N20_FIRST_FULL = 201804      # NOAA-20 archive starts 1 Apr 2018
MIN_MODIS_TOTAL = 200        # below this pooled Aqua total, a monthly ratio is unstable

OUT_COLS = ["cell_id", "cell_lat", "cell_lon", "year", "month", "raw_modis",
            "raw_viirs", "harmonized_count", "harmonized_celldays"]
RATIO_COLS = ["month", "ratio_count", "ratio_celldays", "fit_years", "kind"]


# ----------------------------------------------------------------- loading
def load_cell_days(path):
    """Read cell_days and keep only the three core satellites."""
    cols = ["cell_id", "cell_lat", "cell_lon", "date", "satellite", "n_hotspots"]
    cd = pd.read_parquet(path, columns=cols)
    before = len(cd)
    cd = cd[cd["satellite"].isin(CORE_SATS)].copy()
    print(f"cell_days: {before:,} rows, {len(cd):,} after keeping {CORE_SATS}")
    date = pd.to_datetime(cd["date"])
    cd["year"] = date.dt.year.astype("int16")
    cd["month"] = date.dt.month.astype("int8")
    cd["date"] = date
    return cd


def monthly_wide(cd):
    """One row per cell-month, with count_<sat> and celldays_<sat> columns."""
    g = (cd.groupby(["cell_id", "year", "month", "satellite"], observed=True)
           .agg(count=("n_hotspots", "sum"), celldays=("n_hotspots", "size")))
    wide = g.unstack("satellite", fill_value=0)
    wide.columns = [f"{metric}_{sat}" for metric, sat in wide.columns]
    for metric in ("count", "celldays"):           # make sure every column exists
        for sat in CORE_SATS:
            wide[f"{metric}_{sat}"] = wide.get(f"{metric}_{sat}", 0)
    wide = wide.reset_index()
    wide["ym"] = wide["year"].astype(int) * 100 + wide["month"].astype(int)
    return wide


# ------------------------------------------------------------------ ratios
def modis_viirs_ratios(wide):
    """r_m = sum S-NPP / sum Aqua per calendar month, pooled India, 2012-2018."""
    fit = wide[wide["year"].between(FIT_START, FIT_END) & (wide["ym"] >= SNPP_FIRST_FULL)]
    by_m = fit.groupby("month")[["count_S-NPP", "count_Aqua",
                                 "celldays_S-NPP", "celldays_Aqua"]].sum()
    annual_c = by_m["count_S-NPP"].sum() / by_m["count_Aqua"].sum()
    annual_d = by_m["celldays_S-NPP"].sum() / by_m["celldays_Aqua"].sum()

    rows = []
    for m in range(1, 13):
        s = by_m.loc[m] if m in by_m.index else None
        aqua_total = 0 if s is None else s["count_Aqua"]
        if s is None or aqua_total < MIN_MODIS_TOTAL or s["celldays_Aqua"] == 0:
            rc, rd, note = annual_c, annual_d, f"{FIT_START}-{FIT_END} annual_fallback"
        else:
            rc = s["count_S-NPP"] / s["count_Aqua"]
            rd = s["celldays_S-NPP"] / s["celldays_Aqua"]
            note = f"{FIT_START}-{FIT_END}"
        rows.append({"month": m, "ratio_count": rc, "ratio_celldays": rd,
                     "fit_years": note, "kind": "modis_viirs"})
        print(f"  month {m:2d}: Aqua total {aqua_total:>9,.0f} -> r_count {rc:6.2f}, "
              f"r_celldays {rd:6.2f}  [{note}]")
    print(f"  pooled annual: r_count {annual_c:.2f}, r_celldays {annual_d:.2f}")
    return pd.DataFrame(rows)


def n20_snpp_ratios(wide, switch_ym):
    """NOAA-20 / S-NPP per calendar month on the overlap (full months only)."""
    ov = wide[(wide["ym"] >= N20_FIRST_FULL) & (wide["ym"] < switch_ym)]
    by_m = ov.groupby("month")[["count_NOAA-20", "count_S-NPP",
                                "celldays_NOAA-20", "celldays_S-NPP"]].sum()
    pooled_c = by_m["count_NOAA-20"].sum() / by_m["count_S-NPP"].sum()
    pooled_d = by_m["celldays_NOAA-20"].sum() / by_m["celldays_S-NPP"].sum()
    last_year = (switch_ym - 1) // 100 if switch_ym % 100 != 1 else switch_ym // 100 - 1

    rows = []
    for m in range(1, 13):
        if m in by_m.index and by_m.loc[m, "count_S-NPP"] > 0:
            s = by_m.loc[m]
            rc = s["count_NOAA-20"] / s["count_S-NPP"]
            rd = s["celldays_NOAA-20"] / s["celldays_S-NPP"]
        else:
            rc, rd = pooled_c, pooled_d
        rows.append({"month": m, "ratio_count": rc, "ratio_celldays": rd,
                     "fit_years": f"2018-{last_year}", "kind": "n20_snpp"})
        print(f"  month {m:2d}: NOAA-20/S-NPP count {rc:5.3f}, celldays {rd:5.3f}")
    print(f"  pooled: count {pooled_c:.3f}, celldays {pooled_d:.3f}  (expect about 1)")
    return pd.DataFrame(rows)


# --------------------------------------------------------------- harmonize
def build_cells_monthly(wide, coords, r_mv, r_n20, switch_ym, modis_last_ym):
    ym, m = wide["ym"], wide["month"]
    rc = m.map(r_mv.set_index("month")["ratio_count"])
    rd = m.map(r_mv.set_index("month")["ratio_celldays"])
    nc = m.map(r_n20.set_index("month")["ratio_count"])
    nd = m.map(r_n20.set_index("month")["ratio_celldays"])

    pre = ym < SNPP_FIRST_FULL                     # MODIS era (incl. partial Jan 2012)
    snpp = (ym >= SNPP_FIRST_FULL) & (ym < switch_ym)
    n20 = ym >= switch_ym

    out = wide[["cell_id", "year", "month"]].copy()

    # RAW columns (unscaled). NaN = that sensor had no data in that month.
    modis_ok = ym <= modis_last_ym
    out["raw_modis"] = wide["count_Aqua"].where(modis_ok).astype("float32")
    raw_v = np.select([ym < 201201, ym < switch_ym],
                      [np.nan, wide["count_S-NPP"]], default=wide["count_NOAA-20"])
    out["raw_viirs"] = pd.Series(raw_v, index=wide.index).astype("float32")

    # HARMONIZED columns, all on the S-NPP VIIRS scale
    out["harmonized_count"] = np.select(
        [pre, snpp, n20],
        [wide["count_Aqua"] * rc, wide["count_S-NPP"], wide["count_NOAA-20"] / nc])
    hd = np.select(
        [pre, snpp, n20],
        [wide["celldays_Aqua"] * rd, wide["celldays_S-NPP"], wide["celldays_NOAA-20"] / nd])
    # A cell cannot be active on more days than the month has
    days_in_month = pd.to_datetime(dict(year=wide["year"], month=wide["month"], day=1)).dt.days_in_month
    clipped = int((hd > days_in_month).sum())
    out["harmonized_celldays"] = np.minimum(hd, days_in_month)
    out["harmonized_count"] = out["harmonized_count"].astype("float32")
    out["harmonized_celldays"] = out["harmonized_celldays"].astype("float32")
    print(f"  harmonized_celldays clipped to days-in-month on {clipped:,} cell-months")

    out = out.merge(coords, on="cell_id", how="left")
    out["year"] = out["year"].astype("int16")
    out["month"] = out["month"].astype("int8")
    return out[OUT_COLS].sort_values(["cell_id", "year", "month"]).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description="FIREWATCH harmonization")
    p.add_argument("--cells", default="data/processed/cell_days.parquet")
    p.add_argument("--out", default="data/processed")
    args = p.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cd = load_cell_days(args.cells)
    coords = cd.drop_duplicates("cell_id")[["cell_id", "cell_lat", "cell_lon"]]

    # Where does each stream end? (decides the switch month and MODIS NaNs)
    # Rule: if S-NPP stops in the first 5 days of a month (2 Nov 2026), that month
    # already belongs to NOAA-20; otherwise NOAA-20 starts the following month.
    last_snpp = cd.loc[cd["satellite"] == "S-NPP", "date"].max()
    if last_snpp.day <= 5:
        switch_ym = last_snpp.year * 100 + last_snpp.month
    else:
        nxt = last_snpp + pd.offsets.MonthBegin(1)
        switch_ym = nxt.year * 100 + nxt.month
    last_aqua = cd.loc[cd["satellite"] == "Aqua", "date"].max()
    modis_last_ym = last_aqua.year * 100 + last_aqua.month
    print(f"S-NPP last date {last_snpp.date()} -> NOAA-20 used from {switch_ym // 100}-{switch_ym % 100:02d}")
    print(f"Aqua last date  {last_aqua.date()}")

    wide = monthly_wide(cd)
    del cd
    print(f"cell-months: {len(wide):,}")

    print("\nMODIS Aqua -> VIIRS S-NPP ratios")
    r_mv = modis_viirs_ratios(wide)
    print("\nNOAA-20 -> S-NPP splice ratios")
    r_n20 = n20_snpp_ratios(wide, switch_ym)
    ratios = pd.concat([r_mv, r_n20], ignore_index=True)[RATIO_COLS]
    ratios["month"] = ratios["month"].astype("int8")
    ratios.to_parquet(out_dir / "ratios.parquet", index=False)

    print("\nBuilding cells_monthly")
    cm = build_cells_monthly(wide, coords, r_mv, r_n20, switch_ym, modis_last_ym)
    cm.to_parquet(out_dir / "cells_monthly.parquet", index=False)

    # ------------------------------------------------------ sanity checks
    print(f"\nSaved ratios.parquet ({len(ratios)} rows) and cells_monthly.parquet "
          f"({len(cm):,} rows, {cm['cell_id'].nunique():,} cells)")
    assert list(cm.columns) == OUT_COLS, "column mismatch"
    assert cm["harmonized_count"].notna().all() and (cm["harmonized_count"] >= 0).all()
    assert cm["cell_lat"].notna().all(), "some cells have no coordinates"
    yearly = cm.groupby("year")[["raw_modis", "raw_viirs", "harmonized_count"]].sum(min_count=1)
    print("\nIndia totals per year (harmonized should have NO jump at 2012):")
    print(yearly.round(0).to_string())


if __name__ == "__main__":
    main()
