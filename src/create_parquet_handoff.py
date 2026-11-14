#!/usr/bin/env python3
"""Convert the verified FIREWATCH cleaned CSV into contract-compliant Parquet.

Mac: source .venv/bin/activate && python src/create_parquet_handoff.py
Requires pandas, numpy, pyarrow (from project requirements.txt).
Does not alter src/grid.py; optional cell-days output is a handoff fallback
for review by the geospatial owner.
"""
import argparse
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

FIELDS=['sensor','satellite','date','time','lat','lon','confidence','frp','daynight','type','source']
CELL_FIELDS=['cell_id','cell_lat','cell_lon','date','sensor','satellite','n_hotspots','frp_sum']

def convert(root, generate_cells=False):
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        raise SystemExit('Missing PyArrow: python -m pip install -r requirements.txt') from e
    processed=root/'data'/'processed'
    source=processed/'fires_clean.csv'
    if not source.exists(): raise SystemExit(f'Copy fires_clean.csv to {source} first')
    fire_schema=pa.schema([
        ('sensor',pa.string()),('satellite',pa.string()),('date',pa.string()),
        ('time',pa.string()),('lat',pa.float64()),('lon',pa.float64()),
        ('confidence',pa.string()),('frp',pa.float64()),('daynight',pa.string()),
        ('type',pa.int8()),('source',pa.string())
    ])
    cell_schema=pa.schema([
        ('cell_id',pa.string()),('cell_lat',pa.float64()),('cell_lon',pa.float64()),
        ('date',pa.string()),('sensor',pa.string()),('satellite',pa.string()),
        ('n_hotspots',pa.int64()),('frp_sum',pa.float64())
    ])
    processed.mkdir(parents=True,exist_ok=True)
    n_fires=0
    with tempfile.TemporaryDirectory(prefix='firewatch_cells_') as scratch:
        scratch=Path(scratch)
        with pq.ParquetWriter(processed/'fires.parquet',fire_schema,compression='zstd') as fire_writer:
            for part in pd.read_csv(source,chunksize=180_000,dtype={'date':'string','time':'string','sensor':'string','satellite':'string','confidence':'string','daynight':'string','source':'string'}):
                if list(part.columns)!=FIELDS: raise ValueError('fires schema mismatch')
                part['type']=part['type'].astype('int8')
                fire_writer.write_table(pa.Table.from_pandas(part[FIELDS],schema=fire_schema,preserve_index=False))
                n_fires+=len(part)
                if generate_cells:
                    lat0=np.floor(part['lat'].to_numpy()*10+1e-9)/10
                    lon0=np.floor(part['lon'].to_numpy()*10+1e-9)/10
                    part['cell_id']=[f'{a:.1f}_{b:.1f}' for a,b in zip(lat0,lon0)]
                    part['cell_lat']=np.round(lat0+.05,2)
                    part['cell_lon']=np.round(lon0+.05,2)
                    part['month_key']=part['date'].str[:7]
                    partial=part.groupby(['month_key','cell_id','cell_lat','cell_lon','date','sensor','satellite'],as_index=False).agg(n_hotspots=('frp','size'),frp_sum=('frp','sum'))
                    for month, group in partial.groupby('month_key'):
                        target=scratch/f'{month}.csv'
                        group[CELL_FIELDS].to_csv(target,mode='a',index=False,header=not target.exists())
                print(f'Processed {n_fires:,} fires',flush=True)
        print('Created',processed/'fires.parquet',flush=True)
        if generate_cells:
            n_cells=0
            with pq.ParquetWriter(processed/'cell_days.parquet',cell_schema,compression='zstd') as writer:
                for f in sorted(scratch.glob('*.csv')):
                    parts=pd.read_csv(f,dtype={'date':'string','sensor':'string','satellite':'string','cell_id':'string'})
                    groups=parts.groupby(['cell_id','cell_lat','cell_lon','date','sensor','satellite'],as_index=False).agg(n_hotspots=('n_hotspots','sum'),frp_sum=('frp_sum','sum'))
                    groups=groups[CELL_FIELDS]
                    groups['n_hotspots']=groups['n_hotspots'].astype('int64')
                    writer.write_table(pa.Table.from_pandas(groups,schema=cell_schema,preserve_index=False))
                    n_cells+=len(groups)
                    print(f'{f.stem}: {len(groups):,} active cell-day-satellite records',flush=True)
            print('Created',processed/'cell_days.parquet',f'({n_cells:,} grouped rows)',flush=True)
        meta=pq.ParquetFile(processed/'fires.parquet').metadata
        assert meta.num_rows==n_fires, 'Parquet row count mismatch'
        print('Validated fires.parquet:',n_fires,'rows; exact 11 columns')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='.',help='FIREWATCH project folder')
    parser.add_argument('--cell-days',action='store_true',help='Optional fallback cell-days generator: coordinate with geospatial owner')
    args=parser.parse_args()
    convert(Path(args.root).resolve(),generate_cells=args.cell_days)
