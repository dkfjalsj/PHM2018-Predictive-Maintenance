from pathlib import Path
from src.config import PROJECT_ROOT, data_root, output_dir, report_workdir
from collections import Counter
import pandas as pd
import numpy as np
import json, hashlib, html, time, re

ROOT=data_root()/'train'
OUT=output_dir('sensor_validation')
CHUNK=100000; H=3600; REPS=['03M02','09M02','06M02']
prior=PROJECT_ROOT/'reports'/'fault_analysis'
baseline=json.loads((prior/'quality.json').read_text())
assert baseline['rows']==1236
protected=[p for p in prior.iterdir() if p.is_file()]
oldhash={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
files=sorted(p for p in ROOT.glob('*.csv') if re.fullmatch(r'\d{2}_M\d{2}_DC_train\.csv',p.name)); faults=pd.concat([pd.read_csv(p) for p in sorted((ROOT/'train_faults').glob('*_train_fault_data.csv'))],ignore_index=True)
assert len(faults)==1236 and faults.duplicated().sum()==25
unique=faults.drop_duplicates(['Tool','time','fault_name'])
rows=[]; cols=[]; ttfstats=[]; manifests=[]; samples=[]; events=[]; plotdata={}; mappings=[]
for file in files:
    tool=''.join(file.stem.split('_')[:2]); tf=ROOT/'train_ttf'/file.name
    ff=ROOT/'train_faults'/('_'.join(file.stem.split('_')[:2])+'_train_fault_data.csv')
    assert tf.exists() and ff.exists()
    mappings.append(dict(Tool=tool,train=str(file),train_ttf=str(tf),train_faults=str(ff)))
    meta={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in [file,tf,ff]}
    ffhash=hashlib.sha256(ff.read_bytes()).hexdigest()
    sf=pd.read_csv(file,nrows=0).columns.tolist(); th=pd.read_csv(tf,nrows=0).columns.tolist()
    sensor=sf[7:]; numeric=[x for x in sf if x!='Tool']; stat={c:dict(missing=0,finite=0,nonfinite=0,minimum=np.inf,maximum=-np.inf,noninteger=0,dtypes=set()) for c in numeric}
    n=0; lo=np.inf; hi=-np.inf; prev=None; gaps=Counter(); backwards=0; badtime=0; toolbad=0; mismatch=0; ttfrows=0; dh1=hashlib.sha256();dh2=hashlib.sha256()
    f=unique[unique.Tool.eq(tool)]; et=np.sort(f.time.unique()); ec={int(t):[0,np.inf,-np.inf] for t in et}; tt={}
    pressure=[]; op=[]; binned={c:[[] for _ in range(12)] for c in ['FLOWCOOLPRESSURE','FLOWCOOLFLOWRATE']}
    sensor_reader=pd.read_csv(file,chunksize=CHUNK); ttf_reader=pd.read_csv(tf,chunksize=CHUNK)
    from itertools import zip_longest
    for d,z in zip_longest(sensor_reader,ttf_reader):
        if z is not None:
            ttfrows+=len(z); zt=z.time.to_numpy(dtype=float); dh2.update(zt.astype('<f8').tobytes())
            for c in th[1:]:
                vals=pd.to_numeric(z[c],errors='coerce').to_numpy(dtype=float); fault=c.removeprefix('TTF_'); ft=np.sort(f.loc[f.fault_name.eq(fault),'time'].unique())
                idx=np.searchsorted(ft,zt,side='left'); nxt=np.full(len(zt),np.nan); ok=idx<len(ft); nxt[ok]=ft[idx[ok]]
                idxr=np.searchsorted(ft,zt,side='right'); nxtr=np.full(len(zt),np.nan); okr=idxr<len(ft); nxtr[okr]=ft[idxr[okr]]
                q=tt.setdefault(c,Counter()); finite=np.isfinite(vals); expected=np.isfinite(nxt)
                q.update(rows=len(vals),missing=int(np.isnan(vals).sum()),infinite=int(np.isinf(vals).sum()),negative=int((vals<0).sum()),zero=int((vals==0).sum()),finite=int(finite.sum()),expected_next=int(expected.sum()),match_ge=int((finite & expected & (vals==nxt-zt)).sum()),match_gt=int((finite & np.isfinite(nxtr) & (vals==nxtr-zt)).sum()),unexpected_finite_after_last=int((finite & ~expected).sum()),missing_with_future=int((~finite & expected).sum()),mismatch_ge=int((finite & expected & (vals!=nxt-zt)).sum()))
        if d is None: mismatch+=len(z); continue
        t=d.time.to_numpy(dtype=float); dh1.update(t.astype('<f8').tobytes()); badtime+=int((~np.isfinite(t)).sum()); toolbad+=int(d.Tool.ne(tool).sum())
        if z is None: mismatch+=len(d)
        else:
            k=min(len(d),len(z)); mismatch+=int(np.count_nonzero(t[:k]!=zt[:k]))+abs(len(d)-len(z))
        if not np.isfinite(t).all(): raise ValueError('Invalid sensor time: '+tool)
        dif=np.diff(t if prev is None else np.r_[prev,t]); gaps.update(dif.tolist()); backwards+=int((dif<0).sum()); prev=t[-1]
        lo=min(lo,float(t.min())); hi=max(hi,float(t.max()))
        for c in numeric:
            a=pd.to_numeric(d[c],errors='coerce').to_numpy(dtype=float); q=stat[c]; finite=np.isfinite(a); q['dtypes'].add(str(d[c].dtype)); q['missing']+=int(d[c].isna().sum()); q['nonfinite']+=int((~finite & ~np.isnan(a)).sum()); q['finite']+=int(finite.sum())
            if finite.any(): q['minimum']=min(q['minimum'],float(a[finite].min())); q['maximum']=max(q['maximum'],float(a[finite].max())); q['noninteger']+=int((a[finite]%1!=0).sum())
        # Deterministic systematic sample; full scan counts/min/max above are exact.
        pick=(np.arange(len(d))+n)%200==0
        sm=d.loc[pick,sensor+['stage','recipe','recipe_step']].copy(); sm['Tool']=tool; samples.append(sm)
        # Each event is checked against its own strictly preceding window, even if windows overlap.
        for ev in et[(et>t.min()) & (et<=t.max()+H)]:
            mask=(t>=ev-H)&(t<ev); a=t[mask]
            if len(a): q=ec[int(ev)]; q[0]+=len(a); q[1]=min(q[1],float(a.min())); q[2]=max(q[2],float(a.max()))
        if tool in REPS:
            ni=np.searchsorted(et,t,side='right'); future=np.full(len(t),np.inf); ok=ni<len(et); future[ok]=et[ni[ok]]; delta=future-t
            pi=np.searchsorted(et,t,side='right')-1; past=np.full(len(t),-np.inf); ok=pi>=0; past[ok]=et[pi[ok]]
            pre=(delta>0)&(delta<=H); general=(delta>H)&((t-past)>H)
            # Sampling for display only; timing and TTF checks use all rows.
            pp=d.loc[pick & pre,['FLOWCOOLPRESSURE','FLOWCOOLFLOWRATE','stage','recipe','recipe_step']].copy(); pp['relative_time']=-delta[pick & pre]; pressure.append(pp)
            oo=d.loc[pick & general,['FLOWCOOLPRESSURE','FLOWCOOLFLOWRATE','stage','recipe','recipe_step']].copy(); op.append(oo)
        n+=len(d)
    def quant(counter,p):
        if not counter:return None
        target=p*(sum(counter.values())-1); a=int(np.floor(target));b=int(np.ceil(target)); total=0; va=vb=None
        for value,count in sorted(counter.items()):
            if total<=a<total+count:va=value
            if total<=b<total+count:vb=value;break
            total+=count
        return va+(vb-va)*(target-a)
    rows.append(dict(Tool=tool,rows=n,first_time=lo,last_time=hi,interval_n=sum(gaps.values()),interval_min=min(gaps),interval_median=quant(gaps,.5),interval_p95=quant(gaps,.95),interval_max=max(gaps),interval_mean=sum(k*v for k,v in gaps.items())/sum(gaps.values()),zero_intervals=gaps[0],backward_intervals=backwards,invalid_time=badtime,tool_mismatch=toolbad,sensor_missing_cells=sum(stat[c]['missing'] for c in sensor),sensor_cells=n*len(sensor),sensor_missing_pct=100*sum(stat[c]['missing'] for c in sensor)/(n*len(sensor)),ttf_rows=ttfrows,time_row_mismatch=mismatch,time_sequence_hash_equal=dh1.hexdigest()==dh2.hexdigest()))
    for c,q in stat.items():
        cols.append(dict(Tool=tool,column=c,role='sensor' if c in sensor else 'metadata',rows=n,dtypes='|'.join(sorted(q['dtypes'])),missing=q['missing'],missing_pct=100*q['missing']/n,nonfinite=q['nonfinite'],finite=q['finite'],minimum=q['minimum'] if q['finite'] else None,maximum=q['maximum'] if q['finite'] else None,constant_finite=q['finite']>0 and q['minimum']==q['maximum'],noninteger=q['noninteger']))
    for c,q in tt.items():ttfstats.append(dict(Tool=tool,column=c,**q))
    for _,event in f.iterrows():
        count,emin,emax=ec[int(event.time)]; events.append(dict(Tool=tool,fault_name=event.fault_name,time=int(event.time),in_range=lo<=event.time<=hi,pre_rows=count,pre_span=emax-emin if count else 0,window_start_in_range=event.time-H>=lo,window_coverage=count>=30 and emax-emin>=.8*H))
    if tool in REPS:
        a=pd.concat(pressure,ignore_index=True) if pressure else pd.DataFrame(); b=pd.concat(op,ignore_index=True) if op else pd.DataFrame(); plotdata[tool]=(a,b)
        a.to_csv(OUT/(tool+'_pre_window_sample.csv'),index=False); b.to_csv(OUT/(tool+'_general_sample.csv'),index=False)
    for p in [file,tf,ff]:
        assert meta[str(p)]==(p.stat().st_size,p.stat().st_mtime_ns)
        manifests.append(dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,unchanged_size_mtime=True))
    assert hashlib.sha256(ff.read_bytes()).hexdigest()==ffhash
    print(tool,n,'time_mismatch',mismatch,'TTF mismatches',sum(v['mismatch_ge'] for v in tt.values()),flush=True)
    # Per-tool checkpoints permit review without retaining full sensor tables.
    pd.DataFrame(rows).to_csv(OUT/'equipment_validation.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(cols).to_csv(OUT/'column_quality_by_tool.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(ttfstats).to_csv(OUT/'ttf_consistency.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(mappings).to_csv(OUT/'file_mapping.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(manifests).to_csv(OUT/'source_metadata_manifest.csv',index=False,encoding='utf-8-sig')
ev=pd.DataFrame(events); ev.to_csv(OUT/'event_coverage.csv',index=False,encoding='utf-8-sig')
sample=pd.concat(samples,ignore_index=True); sample.to_pickle(OUT/'distribution_sample.pkl')
summ=[]
for c in sensor:
    a=pd.to_numeric(sample[c],errors='coerce'); a=a[np.isfinite(a)]; q1,q3=a.quantile([.25,.75]); iqr=q3-q1
    summ.append(dict(column=c,sample_n=len(a),q01=a.quantile(.01),q25=q1,median=a.median(),q75=q3,q99=a.quantile(.99),iqr=iqr,lower_fence=q1-1.5*iqr,upper_fence=q3+1.5*iqr,outside_iqr=int(((a<q1-1.5*iqr)|(a>q3+1.5*iqr)).sum()),outside_pct=100*((a<q1-1.5*iqr)|(a>q3+1.5*iqr)).mean()))
pd.DataFrame(summ).to_csv(OUT/'sensor_sample_distributions.csv',index=False,encoding='utf-8-sig')
counts=[]
for fault,g in faults.groupby('fault_name'):
    u=ev[ev.fault_name.eq(fault)]; base=dict(fault_name=fault,raw_rows=len(g),unique_records=len(u),duplicate_excess=len(g)-len(u),outside_range=int((~u.in_range).sum()),in_range=int(u.in_range.sum()),prewindow_coverage=int((u.in_range&u.window_start_in_range&u.window_coverage).sum()))
    for threshold in [0,60,600,3600]:
        clusters=0; eligible=0
        for tool,gg in u.groupby('Tool'):
            gg=gg.sort_values('time'); groups=(gg.time.diff().fillna(np.inf)>threshold).cumsum()
            for _,cluster in gg.groupby(groups):
                # Anchor earliest record; no later event can replace missing prehistory.
                anchor=cluster.iloc[0];clusters+=1; eligible+=int(anchor.in_range and anchor.window_start_in_range and anchor.window_coverage)
        counts.append(dict(**base,proximity_threshold=threshold,clusters=clusters,eligible_cluster_anchors=eligible))
pd.DataFrame(counts).to_csv(OUT/'event_count_sensitivity.csv',index=False,encoding='utf-8-sig')
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in oldhash.items())
(OUT/'validation_summary.json').write_text(json.dumps(dict(sensor_rows=sum(r['rows'] for r in rows),tools=len(rows),chunk_size=CHUNK,window=H,representatives=REPS,old_outputs_hash_unchanged=True),indent=2))
print('SCAN COMPLETE',flush=True)
