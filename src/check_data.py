#!/usr/bin/env python3
"""FIREWATCH v3: validate actual cleaned Parquet handoff (no fabricated counts).

Run from repository root:
    python src/check_data.py
    python src/check_data.py --expected-fires 11503467   # historical-only run

Uses PyArrow and pandas already specified by FIREWATCH requirements.txt.
Writes reports/check_data.json and reports/rows_per_year_satellite.csv.
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd

FIRE_COLS = ['sensor','satellite','date','time','lat','lon','confidence','frp','daynight','type','source']
CELL_COLS = ['cell_id','cell_lat','cell_lon','date','sensor','satellite','n_hotspots','frp_sum']
SATELLITES = {'Aqua', 'S-NPP', 'NOAA-20', 'NOAA-21'}


def main():
    parser = argparse.ArgumentParser(description='Check FIREWATCH Parquet outputs before handoff')
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--expected-fires', type=int, default=None,
                        help='Optional exact count for a known release; omit after adding NRT')
    args = parser.parse_args()
    root = args.root.resolve()
    processed = root / 'data' / 'processed'
    report_dir = root / 'reports'
    report_dir.mkdir(exist_ok=True, parents=True)
    checks = {}
    statistics = {'files': {}, 'fire_rows_by_satellite': {}, 'latest_date_per_satellite': {},
                  'rows_per_year_satellite': [], 'sources': {}}

    def check(key, ok, detail=''):
        checks[key] = {'status': 'PASS' if ok else 'FAIL', 'detail': str(detail)}
        print(f"{'PASS' if ok else 'FAIL'}: {key}" + (f' — {detail}' if detail else ''), flush=True)

    try:
        import pyarrow.parquet as pq
    except ImportError:
        print('ERROR: PyArrow missing. Activate .venv and run python -m pip install -r requirements.txt')
        return 2

    fire_path = processed / 'fires.parquet'
    cell_path = processed / 'cell_days.parquet'
    for label, path in [('fires', fire_path), ('cell_days', cell_path)]:
        check(label + '_file_exists', path.exists(), str(path))
    if not fire_path.exists() or not cell_path.exists():
        print('Both Parquet files are required; run python src/create_parquet_handoff.py --cell-days')
        return 2

    fire = pq.ParquetFile(fire_path)
    cells = pq.ParquetFile(cell_path)
    statistics['files'] = {
        'fires.parquet': {'rows': fire.metadata.num_rows, 'bytes': fire_path.stat().st_size},
        'cell_days.parquet': {'rows': cells.metadata.num_rows, 'bytes': cell_path.stat().st_size}}
    check('fires_schema_exact', fire.schema_arrow.names == FIRE_COLS, fire.schema_arrow.names)
    check('cell_days_schema_exact', cells.schema_arrow.names == CELL_COLS, cells.schema_arrow.names)
    if fire.schema_arrow.names != FIRE_COLS or cells.schema_arrow.names != CELL_COLS:
        return 2

    fire_total = 0
    invalid = Counter()
    by_year_sat = Counter()
    by_year_sat_cell = Counter()
    by_sat = Counter()
    by_source = Counter()
    last_date = {}
    provisional_months = Counter()

    for batch in fire.iter_batches(batch_size=180_000):
        df = batch.to_pandas()
        n = len(df)
        fire_total += n
        invalid['null_fields'] += int(df[FIRE_COLS].isna().any(axis=1).sum())
        invalid['type_not_0'] += int((df['type'] != 0).sum())
        invalid['low_confidence'] += int((~df['confidence'].isin(['nominal', 'high'])).sum())
        invalid['bad_sensor_satellite'] += int((
            ~((df['sensor'].eq('MODIS') & df['satellite'].eq('Aqua')) |
              (df['sensor'].eq('VIIRS') & df['satellite'].isin(['S-NPP','NOAA-20','NOAA-21'])))
        ).sum())
        invalid['bad_location'] += int((~df.lat.between(6, 38.5) | ~df.lon.between(68, 98)).sum())
        invalid['bad_source'] += int((~df.source.isin(['archive', 'nrt'])).sum())
        invalid['bad_frp'] += int((~pd.to_numeric(df.frp, errors='coerce').ge(0)).sum())
        dt = pd.to_datetime(df.date, errors='coerce', format='%Y-%m-%d')
        invalid['bad_date'] += int(dt.isna().sum())
        years = dt.dt.year
        invalid['wrong_satellite_year'] += int((
            (df.satellite.eq('Aqua') & ~years.between(2003, 2025)) |
            (df.satellite.eq('S-NPP') & ~dt.between('2012-01-01','2026-11-02')) |
            (df.satellite.eq('NOAA-20') & ~dt.between('2018-01-01','2026-11-14')) |
            (df.satellite.eq('NOAA-21') & ~dt.between('2022-01-01','2026-11-14'))
        ).sum())
        for (year, satellite), c in df.groupby([years, 'satellite'], dropna=False).size().items():
            if pd.notna(year):
                by_year_sat[(int(year), str(satellite))] += int(c)
        for sat, g in df.groupby('satellite'):
            by_sat[str(sat)] += len(g)
            mx = g.date.max()
            if sat not in last_date or mx > last_date[sat]:
                last_date[str(sat)] = str(mx)
        for source, group in df.groupby('source'):
            by_source[str(source)] += len(group)
            if source == 'nrt':
                for month, c in group.date.astype('string').str[:7].value_counts().items():
                    provisional_months[str(month)] += int(c)

    check('fires_row_count_matches_metadata', fire_total == fire.metadata.num_rows,
          f'{fire_total:,} rows')
    if args.expected_fires is not None:
        check('expected_fires_count', fire_total == args.expected_fires,
              f'actual {fire_total:,}; expected {args.expected_fires:,}')
    check('fires_valid', sum(invalid.values()) == 0,
          ', '.join(f'{k}={v}' for k,v in sorted(invalid.items())))
    print('Validating aggregated cell-days...', flush=True)
    cell_total = 0
    active_detections = 0
    invalid_cells = Counter()
    for batch in cells.iter_batches(batch_size=180_000):
        df = batch.to_pandas()
        cell_total += len(df)
        invalid_cells['null_fields'] += int(df[CELL_COLS].isna().any(axis=1).sum())
        invalid_cells['invalid_count'] += int((~df.n_hotspots.ge(1) | (df.n_hotspots % 1 != 0)).sum())
        invalid_cells['bad_frp'] += int((~pd.to_numeric(df.frp_sum, errors='coerce').ge(0)).sum())
        lat0 = (df.cell_lat * 10).round(5)
        lon0 = (df.cell_lon * 10).round(5)
        invalid_cells['bad_cell_centre'] += int(((lat0 % 1 - 0.5).abs() > 1e-5).sum() +
                                                ((lon0 % 1 - 0.5).abs() > 1e-5).sum())
        expected_id = ((df.cell_lat - .05).round(1).map(lambda x: f'{x:.1f}') + '_' +
                       (df.cell_lon - .05).round(1).map(lambda x: f'{x:.1f}'))
        invalid_cells['cell_id_mismatch'] += int((df.cell_id != expected_id).sum())
        invalid_cells['duplicate_keys_within_batch'] += int(df.duplicated(
            subset=['cell_id','date','sensor','satellite']).sum())
        active_detections += int(df.n_hotspots.sum())
        yrs = pd.to_datetime(df.date, errors='coerce', format='%Y-%m-%d').dt.year
        invalid_cells['bad_date'] += int(yrs.isna().sum())
        for (year, sat), val in df.groupby([yrs, 'satellite']).n_hotspots.sum().items():
            if pd.notna(year):
                by_year_sat_cell[(int(year), str(sat))] += int(val)
    check('cell_days_row_count_matches_metadata', cell_total == cells.metadata.num_rows,
          f'{cell_total:,} rows')
    check('cell_day_counts_equal_fires', active_detections == fire_total,
          f'cell-day total={active_detections:,}; fires={fire_total:,}')
    check('year_satellite_aggregation_matches', dict(by_year_sat_cell) == dict(by_year_sat),
          'summed cell-day counts versus source fires, each year and satellite')
    check('cell_days_valid', sum(invalid_cells.values()) == 0,
          ', '.join(f'{k}={v}' for k,v in sorted(invalid_cells.items())))
    statistics['fire_rows_by_satellite'] = dict(sorted(by_sat.items()))
    statistics['latest_date_per_satellite'] = dict(sorted(last_date.items()))
    statistics['sources'] = dict(sorted(by_source.items()))
    statistics['nrt_rows_by_month'] = dict(sorted(provisional_months.items()))
    statistics['rows_per_year_satellite'] = [
        {'year': year, 'satellite': sat, 'fires': n,
         'cell_day_hotspot_sum': by_year_sat_cell.get((year, sat), 0)}
        for (year, sat), n in sorted(by_year_sat.items())]
    pd.DataFrame(statistics['rows_per_year_satellite']).to_csv(
        report_dir/'rows_per_year_satellite.csv', index=False)
    result = {'overall': 'PASS' if all(c['status'] == 'PASS' for c in checks.values()) else 'FAIL',
              'checks': checks, 'statistics': statistics,
              'limitations': [
                  'India bbox is not an administrative polygon and country source must be confirmed.',
                  'NOAA-21 excluded from core harmonization by the science lead.',
                  'If NRT type=0 was inferred, source=nrt is provisional, not observed vegetation classification.',
                  'Full-file global duplicate-key test for cell-days not performed (batch-level test only).',
                  'Unvalidated archive versus NRT overlap and hotspot scientific accuracy require separate review.']}
    out = report_dir/'check_data.json'
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('\nRows per year × satellite:')
    for item in statistics['rows_per_year_satellite']:
        print(f"  {item['year']} {item['satellite']}: {item['fires']:,} fire detections")
    print('\nLatest date per satellite:', statistics['latest_date_per_satellite'])
    print('\nOVERALL:', result['overall'])
    print('Report written:', out)
    return 0 if result['overall'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
