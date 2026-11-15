"""Run at repo root: python scripts/check_integration.py

Read-only verification of the dependencies and real FIREWATCH data products.
No datasets or credentials are modified, and no test metrics are invented.
"""
from pathlib import Path
import importlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
REQUIRED = {
    'cells_monthly.parquet': ('cell_id', 'cell_lat', 'cell_lon', 'year', 'month',
                               'raw_modis', 'raw_viirs', 'harmonized_count', 'harmonized_celldays'),
    'fires.parquet': ('sensor', 'satellite', 'date', 'time', 'lat', 'lon', 'confidence',
                      'frp', 'daynight', 'type', 'source'),
}

def main() -> int:
    errors, warnings = [], []
    for mod in ('streamlit', 'pandas', 'pyarrow', 'plotly', 'folium', 'streamlit_folium'):
        try:
            importlib.import_module(mod)
            print('PASS dependency', mod)
        except ImportError as exc:
            errors.append(f'Dependency {mod} not installed: {exc}')
    for name in ('dashboard.py', 'dashboard.html', 'frontend_utils.py'):
        p = ROOT / 'app' / name
        if not p.is_file(): errors.append(f'Missing app/{name}')
        else: print('PASS file', p.relative_to(ROOT))
    for mod in ('data_access', 'geo', 'grid'):
        p = ROOT / 'src' / (mod+'.py')
        if not p.is_file(): errors.append(f'Missing src/{mod}.py')
        else: print('PASS team module', p.relative_to(ROOT))
    if 'pyarrow' in sys.modules:
        import pyarrow.parquet as pq
        for name, columns in REQUIRED.items():
            p = next((x for x in (ROOT/'data'/'processed'/name, ROOT/'data'/'sample'/name) if x.exists()), None)
            if not p:
                errors.append(f'Missing data product {name}; cannot certify real-data integration')
                continue
            try:
                cols = set(pq.read_schema(str(p)).names)
                missing = set(columns) - cols
                if missing:
                    errors.append(f'{name}: missing columns {sorted(missing)}')
                else:
                    print('PASS Parquet schema', name, 'size_MB', round(p.stat().st_size / 1e6, 2))
            except Exception as exc:
                errors.append(f'{name}: unable to read schema: {exc}')
    for name in ('ratios.parquet','validation.json'):
        if not (ROOT/'data'/'processed'/name).exists():
            warnings.append(f'Optional methods/validation product not present: {name}')
    for w in warnings: print('WARNING', w)
    for e in errors: print('FAIL', e)
    if errors:
        print(f'NOT READY — {len(errors)} error(s), {len(warnings)} warning(s)')
        return 1
    print(f'INTEGRATION PREREQUISITES PRESENT — {len(warnings)} warning(s)')
    return 0

if __name__ == '__main__':
    sys.exit(main())
