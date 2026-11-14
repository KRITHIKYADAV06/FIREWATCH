"""Validate FIREWATCH cleaned CSV and build supplemental time-series features.
Usage: python src/validate_handoff.py
Run from the FIREWATCH project root.
"""
import json
from collections import defaultdict
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/processed/fires_clean.csv'
REPORT = ROOT / 'reports'
EXPECTED = ['sensor','satellite','date','time','lat','lon','confidence','frp','daynight','type','source']
counts=defaultdict(lambda: {'hotspot_count':0,'frp_sum':0.0,'high_confidence_count':0,'active_dates':set()})
min_dates={}; max_dates={}; total=0; bad=0; exact_columns=None
for frame in pd.read_csv(SOURCE,chunksize=200_000,dtype={'time':'string','date':'string','sensor':'string','satellite':'string','confidence':'string','daynight':'string','source':'string'}):
    if exact_columns is None:
        exact_columns = list(frame.columns)==EXPECTED
    total += len(frame)
    bad += int((~frame.type.eq(0) | ~frame.confidence.isin(['nominal','high']) | ~frame.sensor.isin(['MODIS','VIIRS']) | ~frame.satellite.isin(['Aqua','S-NPP','NOAA-20','NOAA-21']) | ~frame.lat.between(6,38.5) | ~frame.lon.between(68,98) | frame.date.isna() | ~frame.source.isin(['archive','nrt'])).sum())
    frame['year']=frame.date.str[:4].astype(int)
    frame['month']=frame.date.str[5:7].astype(int)
    for (year,month,satellite,sensor),grp in frame.groupby(['year','month','satellite','sensor']):
        key=(int(year),int(month),str(satellite),str(sensor))
        c=counts[key]
        c['hotspot_count'] += len(grp)
        c['frp_sum'] += float(grp.frp.sum())
        c['high_confidence_count'] += int(grp.confidence.eq('high').sum())
        c['active_dates'].update(grp.date.dropna().unique().tolist())
    for sat,g in frame.groupby('satellite'):
        lo=str(g.date.min());hi=str(g.date.max())
        min_dates[str(sat)]=min(lo,min_dates.get(str(sat),lo))
        max_dates[str(sat)]=max(hi,max_dates.get(str(sat),hi))
rows=[]
for (y,m,sat,sensor),v in sorted(counts.items()):
    rows.append({'year':y,'month':m,'sensor':sensor,'satellite':sat,'hotspot_count':v['hotspot_count'],'frp_sum':round(v['frp_sum'],2),'high_confidence_count':v['high_confidence_count'],'active_dates':len(v['active_dates'])})
REPORT.mkdir(parents=True,exist_ok=True)
pd.DataFrame(rows).to_csv(REPORT/'monthly_satellite_features.csv',index=False)
assert total>0, 'No data rows found'
assert bad==0, f'Quality check failed: {bad} bad rows'
assert exact_columns, 'Schema mismatch'
result={'cleaned_rows':total,'contract_exact':exact_columns,'invalid_contract_rows':bad,'satellite_first_date':min_dates,'satellite_last_date':max_dates,'monthly_feature_rows':len(rows),'approved_satellites':['Aqua','S-NPP','NOAA-20','NOAA-21'],'warnings':['Bounding box is not an India administrative polygon','NRT rows with inferred type=0 must remain labelled provisional; review method with science lead','NOAA-21 is a separate approved stream, do not sum satellites indiscriminately','Active fire hotspots are detections, not independently verified unique fires or burned area']}
(REPORT/'handoff_validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
