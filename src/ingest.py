#!/usr/bin/env python3
"""FIREWATCH v3: stream archive CSVs from ZIPs into an exact-schema Parquet.

Examples:
  python src/ingest.py --inputs 'MODIS.zip' 'Data.zip' 'VIIRS.zip' '2026 dataset.zip' --report-only
  python src/ingest.py --inputs 'MODIS.zip' 'Data.zip' 'VIIRS.zip' '2026 dataset.zip' --output-dir data/processed --sample-dir data/sample

Use archive sources only (source='archive'). NRT rows without `type` are
excluded rather than being silently classified as vegetation fires.
Core policy: MODIS Aqua 2003-2025, VIIRS SNPP 2012-2026-11-02,
VIIRS NOAA20 2018+. NOAA-21 approved by lead; included as a distinct VIIRS satellite. MODIS 2026 excluded per core plan.
"""
import argparse
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ['sensor','satellite','date','time','lat','lon','confidence','frp','daynight','type','source']
BASE_FIELDS = {'latitude','longitude','acq_date','acq_time','satellite','confidence','frp','daynight','type'}
# Broad bounding box, NOT precise Indian administrative boundaries.
BBOX = (6.0, 38.5, 68.0, 98.0)
SAMPLE_BBOX = (27.6, 32.6, 73.8, 77.6)
SAMPLE_START, SAMPLE_END = '2023-11-01', '2023-11-30'


def get_source(filename):
    if 'nrt' in filename.lower():
        return 'nrt'
    return 'archive'


def selected_sensor(filename, header):
    name = filename.lower()
    if 'j2v-' in name:
        return 'NOAA-21'
    if 'j1v-' in name or 'jpss1' in name:
        return 'NOAA-20'
    if 'sv-c2' in name or 'snpp_' in name or 'viirs-snpp' in name:
        return 'S-NPP'
    if 'm-c61' in name or 'modis_' in name:
        return 'MODIS'
    return None


def clean_chunk(frame, filename, stats):
    """Return validated records. Does not invent values in missing fields."""
    original = len(frame)
    stats['input_rows'] += original
    if not BASE_FIELDS.issubset(frame.columns):
        stats['missing_required_rows'] += original
        return pd.DataFrame(columns=COLS)
    # Restrict to planned streams. Note MODIS files can mix Terra and Aqua.
    frame = frame.copy()
    sat = frame['satellite'].astype('string').str.strip().str.upper()
    is_modis = 'MODIS' in filename.upper() or 'M-C61' in filename.upper()
    if is_modis:
        target = sat.eq('AQUA')
        sensor_name, sat_name = 'MODIS', 'Aqua'
    elif 'J2V-' in filename.upper():
        target = sat.isin(['N21', 'NOAA-21'])
        sensor_name, sat_name = 'VIIRS', 'NOAA-21'
    elif 'J1V-' in filename.upper() or 'JPSS1' in filename.upper():
        target = sat.eq('N20') | sat.eq('NOAA-20')
        sensor_name, sat_name = 'VIIRS', 'NOAA-20'
    elif 'SV-C2' in filename.upper() or 'SNPP' in filename.upper():
        target = sat.isin(['N','SNPP','S-NPP'])
        sensor_name, sat_name = 'VIIRS', 'S-NPP'
    else:
        stats['unknown_source_rows'] += original
        return pd.DataFrame(columns=COLS)
    stats['other_satellite_rows'] += int((~target.fillna(False)).sum())
    frame = frame.loc[target.fillna(False)].copy()
    if frame.empty:
        return pd.DataFrame(columns=COLS)

    date = pd.to_datetime(frame['acq_date'], errors='coerce')
    if sensor_name == 'MODIS':
        date_ok = date.between('2003-01-01','2025-12-31')
    elif sat_name == 'S-NPP':
        date_ok = date.between('2012-01-01','2026-11-02')
    elif sat_name == 'NOAA-21':
        date_ok = date.between('2022-01-01', '2026-11-14')
    else:
        date_ok = date.between('2018-01-01', '2026-11-14')
    stats['outside_time_window'] += int((~date_ok).sum())
    frame = frame.loc[date_ok].copy()
    date = date.loc[date_ok]
    if frame.empty:
        return pd.DataFrame(columns=COLS)

    fire_type = pd.to_numeric(frame['type'], errors='coerce')
    is_fire = fire_type.eq(0)
    stats['non_vegetation_or_missing_type'] += int((~is_fire).sum())
    frame = frame.loc[is_fire].copy()
    date = date.loc[is_fire]
    if frame.empty:
        return pd.DataFrame(columns=COLS)

    conf_raw = frame['confidence'].astype('string').str.strip().str.lower()
    if sensor_name == 'MODIS':
        conf_num = pd.to_numeric(conf_raw, errors='coerce')
        conf = pd.Series(np.select([conf_num.lt(30),conf_num.between(30,79),conf_num.ge(80)], ['low','nominal','high'], default='invalid'),index=frame.index)
    else:
        conf = conf_raw.map({'l':'low','n':'nominal','h':'high','low':'low','nominal':'nominal','high':'high'}).fillna('invalid')
    keep = conf.isin(['nominal','high'])
    stats['low_or_invalid_confidence'] += int((~keep).sum())
    frame = frame.loc[keep].copy()
    date = date.loc[keep]
    conf = conf.loc[keep]
    if frame.empty:
        return pd.DataFrame(columns=COLS)

    lat = pd.to_numeric(frame['latitude'], errors='coerce')
    lon = pd.to_numeric(frame['longitude'], errors='coerce')
    frp = pd.to_numeric(frame['frp'], errors='coerce')
    t = frame['acq_time'].astype('string').str.replace(r'\.0$', '',regex=True).str.zfill(4)
    # Prevent malformed HHMM fields from silently passing.
    valid_time = t.str.fullmatch(r'\d{4}').fillna(False) & t.str.slice(0,2).astype('string').str.isnumeric().fillna(False)
    hour = pd.to_numeric(t.str[:2],errors='coerce')
    minute = pd.to_numeric(t.str[2:],errors='coerce')
    valid_time = valid_time & hour.between(0,23) & minute.between(0,59)
    dn = frame['daynight'].astype('string').str.upper().str.strip()
    valid = (lat.between(BBOX[0],BBOX[1]) & lon.between(BBOX[2],BBOX[3]) &
             frp.notna() & frp.ge(0) & valid_time & dn.isin(['D','N']))
    stats['bad_coordinates_or_fields'] += int((~valid).sum())
    frame = frame.loc[valid]
    if frame.empty:
        return pd.DataFrame(columns=COLS)
    return pd.DataFrame({
        'sensor':sensor_name,'satellite':sat_name,'date':date.loc[valid].dt.strftime('%Y-%m-%d').to_numpy(),
        'time':t.loc[valid].to_numpy(),'lat':lat.loc[valid].to_numpy(), 'lon':lon.loc[valid].to_numpy(),
        'confidence':conf.loc[valid].to_numpy(),'frp':frp.loc[valid].to_numpy(),
        'daynight':dn.loc[valid].to_numpy(),'type':np.int8(0),'source':get_source(filename),
    },columns=COLS)


def run(args):
    stats = Counter()
    coverage = Counter()
    source_counts = Counter()
    file_reports=[]
    seen_hashes=set()
    sample_frames=[]
    writer=None
    csv_written=False
    pa=pq=None
    if args.clean_csv:
        Path(args.clean_csv).parent.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as e:
            raise SystemExit('PyArrow missing. Activate your Mac .venv and run: python -m pip install pyarrow') from e
        Path(args.output_dir).mkdir(parents=True,exist_ok=True)
        Path(args.sample_dir).mkdir(parents=True,exist_ok=True)

    # Archive before NRT, to prefer archive when duplicates are detected.
    entries=[]
    for path in args.inputs:
        with zipfile.ZipFile(path) as z:
            for item in z.infolist():
                if not item.filename.lower().endswith('.csv') or item.filename.startswith('__MACOSX/'):
                    continue
                entries.append((path,item.filename,item.file_size))
    entries.sort(key=lambda t:('nrt' in t[1].lower(),t[1]))
    for outer,inner,size in entries:
        file_stats=Counter()
        basename=Path(inner).name
        stream=selected_sensor(basename,None)
        if stream is None:
            file_reports.append({'file':basename,'status':'unknown-source-skipped'})
            continue
        if 'nrt' in basename.lower():
            # Per project type==0 rule: NRT has no type. Skip until reviewed.
            file_reports.append({'file':basename,'status':'NRT-skipped-missing-type'})
            continue
        with zipfile.ZipFile(outer) as z, z.open(inner) as fd:
            for raw in pd.read_csv(fd,chunksize=args.chunksize,low_memory=False):
                cleaned=clean_chunk(raw,basename,file_stats)
                if cleaned.empty:
                    continue
                # Deduplicate across archives using a canonical detection key. Source omitted
                # so the same satellite detection from two feeds is only counted once.
                ids=pd.util.hash_pandas_object(cleaned[['sensor','satellite','date','time','lat','lon','frp','daynight','type']],index=False)
                keep = ~ids.duplicated() & ~ids.map(seen_hashes.__contains__)
                file_stats['duplicate_rows'] += int((~keep).sum())
                seen_hashes.update(ids.loc[keep].tolist())
                cleaned=cleaned.loc[keep].reset_index(drop=True)
                if cleaned.empty:
                    continue
                file_stats['kept_rows'] += len(cleaned)
                for (year,sat),count in cleaned.groupby([cleaned['date'].str.slice(0,4),'satellite']).size().items():
                    coverage[f'{year}|{sat}']+=int(count)
                source_counts[cleaned['satellite'].iat[0]]+=len(cleaned)
                in_sample=(cleaned.date.between(SAMPLE_START,SAMPLE_END) &
                          cleaned.lat.between(SAMPLE_BBOX[0],SAMPLE_BBOX[1]) &
                          cleaned.lon.between(SAMPLE_BBOX[2],SAMPLE_BBOX[3]))
                if in_sample.any():
                    sample_frames.append(cleaned.loc[in_sample].copy())
                if args.clean_csv:
                    cleaned.to_csv(args.clean_csv, mode='a', header=not csv_written, index=False)
                    csv_written=True
                if not args.report_only:
                    # Consistent Arrow schema from the first batch is sufficient here.
                    table=pa.Table.from_pandas(cleaned,preserve_index=False)
                    if writer is None:
                        writer=pq.ParquetWriter(str(Path(args.output_dir)/'fires.parquet'),table.schema,compression='zstd')
                    writer.write_table(table)
        stats.update(file_stats)
        file_reports.append({'file':basename,'status':'processed',**dict(file_stats)})
        print(f'{basename}: {file_stats["input_rows"]:,} input -> {file_stats["kept_rows"]:,} kept',flush=True)
    if writer is not None:
        writer.close()

    report={
        'report_type':'FIREWATCH ingestion quality report',
        'policy':'MODIS Aqua 2003-2025; VIIRS S-NPP through 2026-11-02; NOAA-20 2018+; NOAA-21 2022+ (lead-approved); type=0; nominal/high confidence; broad India bbox',
        'input_archives':[Path(x).name for x in args.inputs],
        'source_counts':dict(sorted(source_counts.items())),
        'year_satellite_counts':dict(sorted(coverage.items())),
        'totals':dict(stats),
        'sample_period':f'{SAMPLE_START} through {SAMPLE_END}',
        'sample_bbox':{'min_lat':SAMPLE_BBOX[0],'max_lat':SAMPLE_BBOX[1],'min_lon':SAMPLE_BBOX[2],'max_lon':SAMPLE_BBOX[3]},
        'sample_rows':int(sum(map(len,sample_frames))),
        'files':file_reports,
        'warnings':[
            'India bounding box is NOT an administrative India polygon; border-country detections may remain.',
            '2026 NRT files lack type; excluded pending team-lead approved handling.',
            'NOAA-21 included with team-lead approval, kept as its own satellite; the harmonization algorithm must explicitly select the intended VIIRS reference and avoid summing passes as unique fires.',
            'MODIS Terra excluded: only Aqua is used, per v3 specification.',
            'Counts are FIRMS detections, not unique fires, burned area or forecasts.',
            'NOAA-20 and S-NPP are both retained separately for harmonization; avoid double counting where calendar logic requires a single VIIRS stream.'
        ],
    }
    Path(args.report).parent.mkdir(parents=True,exist_ok=True)
    Path(args.report).write_text(json.dumps(report,indent=2),encoding='utf-8')
    if sample_frames:
        sample=pd.concat(sample_frames,ignore_index=True)
        if args.report_only:
            sample.to_csv(args.sample_csv,index=False)
        else:
            sample.to_parquet(str(Path(args.sample_dir)/'punjab_haryana_2023_11.parquet'),index=False)
    print('TOTAL kept:',sum(source_counts.values()),'SAMPLE:',report['sample_rows'],'REPORT:',args.report)
    return report


def main():
    p=argparse.ArgumentParser(description='FIREWATCH FIRMS archive ingestion')
    p.add_argument('--inputs',nargs='+',required=True,help='Original ZIP files')
    p.add_argument('--output-dir',default='data/processed')
    p.add_argument('--sample-dir',default='data/sample')
    p.add_argument('--report',default='reports/ingestion_quality.json')
    p.add_argument('--clean-csv',default=None,help='Optional streamed cleaned CSV output, works without PyArrow')
    p.add_argument('--sample-csv',default='reports/punjab_haryana_2023_11_sample.csv')
    p.add_argument('--chunksize',type=int,default=80000)
    p.add_argument('--report-only',action='store_true',help='QC and real sample CSV without needing PyArrow')
    run(p.parse_args())

if __name__=='__main__':
    main()
