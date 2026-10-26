"""Stage 3 EDA: bounded-memory stratified reservoirs and causal sensor windows."""
from pathlib import Path
import sys, os, json, hashlib, subprocess, shutil, re
from collections import Counter
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P))
from src.config import data_root
ROOT=data_root()/'train'; LOCAL=P/'.local/eda3'; LOCAL.mkdir(parents=True,exist_ok=True)
R=P/'reports'; F=P/'figures/eda3'; F.mkdir(exist_ok=True)
HORIZONS=[3600,21600,86400,259200,604800]; LABELS=['1h','6h','24h','3d','7d']; W=3600; GAP=60; SEED=2018
KEYS=['stage','recipe','recipe_step']; SENSORS=pd.read_csv(next(ROOT.glob('[0-9][0-9]_M[0-9][0-9]_DC_train.csv')),nrows=0).columns.tolist()[7:]
def save(name,d):d.to_csv(LOCAL/(name+'.csv'),index=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def command(args,**kw):return subprocess.run(list(map(str,args)),check=True,cwd=P,**kw)

def prepare():
    assert not (R/'sensor_profiling_report.html').exists() and not (R/'timeseries_eda_report.html').exists(),'Report already exists; no overwrite'
    old={str(p.relative_to(P)):sha(p) for folder in ['src','reports','figures'] for p in (P/folder).rglob('*') if p.is_file() and '/__pycache__/' not in p.as_posix()}
    (LOCAL/'old_hashes.json').write_text(json.dumps(old))
    for name in ['eda3_stream.py','eda3_reports.py']:
        dest=P/'src'/name
        assert not dest.exists(),dest
        shutil.copy2(Path(__file__).parent/name,dest)
    req=P/'requirements-eda.txt';assert not req.exists()
    req.write_text('ydata-profiling==4.18.4\nsetuptools<82\npandas<3\nnumpy<2.4\n',encoding='utf-8')
    venv=P/'.venv-eda';assert not venv.exists()
    command([sys.executable,'-m','venv',venv])
    py=venv/'Scripts/python.exe'
    command([py,'-m','pip','install','--disable-pip-version-check','-r',req])
    command([py,'-m','pip','check'])
    lock=subprocess.check_output([str(py),'-m','pip','freeze'],text=True)
    (P/'requirements-eda-lock.txt').write_text(lock,encoding='utf-8')

def add_stats(state,a,t):
    if not len(t):return
    finite=np.isfinite(a); x=np.where(finite,a,0.)
    state['n']+=finite.sum(axis=0);state['sum']+=x.sum(axis=0);state['ss']+=(x*x).sum(axis=0)
    state['rows']+=len(t)
    if state['last_t'] is not None:
        delta=t[0]-state['last_t'];state['gap']=max(state['gap'],float(delta));state['duration']+=min(float(delta),4.)
    dif=np.diff(t)
    if len(dif):state['gap']=max(state['gap'],float(dif.max()));state['duration']+=np.minimum(dif,4).sum()
    if state['first_t'] is None:state['first_t']=float(t[0]);state['first']=a[0].copy()
    state['last_t']=float(t[-1]);state['last']=a[-1].copy()
def newstats():return dict(n=np.zeros(17),sum=np.zeros(17),ss=np.zeros(17),rows=0,first_t=None,last_t=None,first=np.full(17,np.nan),last=np.full(17,np.nan),gap=0.,duration=0.,ma_n=np.zeros(17),ma_sum=np.zeros(17),old_n=np.zeros(17),old_sum=np.zeros(17),process={})

def scan():
    allfault=pd.concat([pd.read_csv(p) for p in (ROOT/'train_faults').glob('*_train_fault_data.csv')],ignore_index=True).drop_duplicates(['Tool','time','fault_name'])
    low=allfault[allfault.fault_name.eq('FlowCool Pressure Dropped Below Limit')]
    bounds=pd.read_csv(R/'sensor_validation/equipment_validation.csv').set_index('Tool')
    samplelist=[]; windowrows=[]; prows=[]; controls=[]; hourly_quality=[]; inventory=[]; strata_a=[]; strata_b=[]; cluster_sensitivity=[]
    for file in sorted(ROOT.glob('[0-9][0-9]_M[0-9][0-9]_DC_train.csv')):
        tool=''.join(file.stem.split('_')[:2]); rng=np.random.default_rng(SEED+int(tool.replace('M','')))
        filemeta=(file.stat().st_size,file.stat().st_mtime_ns)
        faults=np.sort(allfault.loc[allfault.Tool.eq(tool),'time'].unique())
        lowtimes=np.sort(low.loc[low.Tool.eq(tool),'time'].unique())
        for threshold in [0,600,3600,21600,86400,604800]:
            cluster_sensitivity.append(dict(Tool=tool,threshold_seconds_assumed=threshold,unique_records=len(lowtimes),clusters=int(np.sum(np.r_[True,np.diff(lowtimes)>threshold])) if len(lowtimes) else 0))
        anchors=lowtimes[np.r_[True,np.diff(lowtimes)>3600]] if len(lowtimes) else lowtimes
        wins=[]
        for event in anchors:
            for h,label in zip(HORIZONS,LABELS):
                end=event-h;start=end-W
                contamination=bool(np.any((faults>=start)&(faults<event)))
                wins.append(dict(Tool=tool,event=int(event),horizon=label,lead_seconds=h,start=start,end=end,contaminated=contamination,s=newstats()))
        ra=None;rb=None; counts_a=Counter();counts_b=Counter(); hourq={};base={}; total=0;dedup=0;prevrow=None;gapcounts=Counter();min_t=float(bounds.loc[tool,'first_time']);max_t=float(bounds.loc[tool,'last_time'])
        for d in pd.read_csv(file,chunksize=100000):
            raw_t=d.time.to_numpy(dtype=float)
            assert np.isfinite(raw_t).all() and np.all(np.diff(raw_t)>=0)
            d['_row']=np.arange(len(d))+total;d['_priority']=rng.random(len(d));d['_quartile']=np.minimum(3,((raw_t-min_t)/(max_t-min_t)*4).astype(int))
            d['_a']=pd.util.hash_pandas_object(d[['recipe','recipe_step','_quartile']],index=False).to_numpy()
            d['_b']=pd.util.hash_pandas_object(d[KEYS],index=False).to_numpy()
            counts_a.update(d['_a'].value_counts().to_dict());counts_b.update(d['_b'].value_counts().to_dict())
            # Independent uniform random priorities preserve a bounded reservoir per stratum.
            ca=d.loc[d.groupby('_a',sort=False)['_priority'].idxmin()]
            cb=d.loc[d.groupby('_b',sort=False)['_priority'].idxmin()]
            ra=ca if ra is None else pd.concat([ra,ca],ignore_index=True).sort_values('_priority').drop_duplicates('_a')
            rb=cb if rb is None else pd.concat([rb,cb],ignore_index=True).sort_values('_priority').drop_duplicates('_b')
            assert len(ra)+len(rb)<100000,'Unexpected stratum count; memory safeguard'
            total+=len(d)
            # For temporal statistics keep the first measurement at each time, including chunk boundaries.
            duplicate=d.time.duplicated().to_numpy(copy=True)
            if prevrow is not None:duplicate[0]|=raw_t[0]==prevrow
            prevrow=raw_t[-1];dedup+=int(duplicate.sum()); d=d.loc[~duplicate].copy()
            t=d.time.to_numpy(dtype=float);x=d[SENSORS].to_numpy(dtype=float)
            if not len(t):continue
            dif=np.diff(t)
            for threshold in [60,300,3600]:gapcounts[threshold]+=int((dif>threshold).sum())
            for w in wins:
                if w['end']<=t[0] or w['start']>t[-1]:continue
                left=np.searchsorted(t,w['start'],'left');right=np.searchsorted(t,w['end'],'left')
                if right<=left:continue
                xx=x[left:right];ts=t[left:right];q=w['s'];add_stats(q,xx,ts)
                for key,lower,upper in [('ma',w['end']-900,w['end']),('old',w['end']-1800,w['end']-900)]:
                    a=xx[(ts>=lower)&(ts<upper)];f=np.isfinite(a);q[key+'_n']+=f.sum(axis=0);q[key+'_sum']+=np.where(f,a,0).sum(axis=0)
                sub=d.iloc[left:right]
                for process,indices in sub.groupby(KEYS,dropna=False,sort=False).indices.items():
                    a=xx[indices];fin=np.isfinite(a);vals=np.where(fin,a,0)
                    old=q['process'].setdefault(process,[np.zeros(17),np.zeros(17)])
                    old[0]+=fin.sum(axis=0);old[1]+=vals.sum(axis=0)
            hours=(t//3600).astype(np.int64);d['_hour']=hours
            for hour,ii in d.groupby('_hour',sort=False).indices.items():
                a=t[ii];q=hourq.setdefault(int(hour),dict(n=0,first=None,last=None,gap=0.,duration=0.))
                if q['last'] is not None:
                    gap=a[0]-q['last'];q['gap']=max(q['gap'],float(gap));q['duration']+=min(float(gap),4.)
                diff=np.diff(a);q['n']+=len(a);q['first']=float(a[0]) if q['first'] is None else q['first'];q['last']=float(a[-1])
                if len(diff):q['gap']=max(q['gap'],float(diff.max()));q['duration']+=np.minimum(diff,4).sum()
            # All-fault exclusion is retrospective for reference construction, never an input variable.
            ip=np.searchsorted(faults,t,'right')-1;ix=np.searchsorted(faults,t,'left');previous=np.full(len(t),-np.inf);future=np.full(len(t),np.inf)
            ok=ip>=0;previous[ok]=faults[ip[ok]];ok=ix<len(faults);future[ok]=faults[ix[ok]]
            general=((t-previous)>604800)&((future-t)>604800)
            sub=d.loc[general]
            if len(sub):
                arr=sub[SENSORS].to_numpy(dtype=float)
                for key,ii in sub.groupby(['_hour']+KEYS,dropna=False,sort=False).indices.items():
                    a=arr[ii];fin=np.isfinite(a);val=np.where(fin,a,0);b=base.setdefault(key,[np.zeros(17),np.zeros(17),np.zeros(17)])
                    b[0]+=fin.sum(axis=0);b[1]+=val.sum(axis=0);b[2]+=(val*val).sum(axis=0)
        assert total==int(bounds.loc[tool,'rows'])
        assert filemeta==(file.stat().st_size,file.stat().st_mtime_ns)
        sm=pd.concat([ra,rb],ignore_index=True).drop_duplicates('_row').copy();sm['Tool']=tool;samplelist.append(sm)
        for key,count in counts_a.items():strata_a.append(dict(Tool=tool,stratum=str(key),population_rows=count,sampled_rows=int((sm._a==key).sum())))
        for key,count in counts_b.items():strata_b.append(dict(Tool=tool,stratum=str(key),population_rows=count,sampled_rows=int((sm._b==key).sum())))
        validhours=set()
        for hour,q in hourq.items():
            start=hour*3600;end=start+3600; coverage=q['duration']/3600
            quiet=not np.any((faults>=start-604800)&(faults<=end+604800))
            good=coverage>=.8 and q['gap']<=60 and q['first']-start<=60 and end-q['last']<=60
            if good and quiet:validhours.add(hour)
            hourly_quality.append(dict(Tool=tool,hour=hour,coverage=coverage,max_gap=q['gap'],quiet_7d=quiet,quality=good))
        for (hour,stage,recipe,step),b in base.items():
            if hour not in validhours:continue
            for j,sensor in enumerate(SENSORS):
                if b[0][j]>=30:controls.append(dict(Tool=tool,hour=int(hour),stage=stage,recipe=recipe,recipe_step=step,sensor=sensor,n=b[0][j],mean=b[1][j]/b[0][j],std=np.sqrt(max(0,(b[2][j]-b[1][j]**2/b[0][j])/(b[0][j]-1)))))
        for w in wins:
            q=w.pop('s');first=q['first_t'];last=q['last_t'];quality=first is not None and q['duration']/3600>=.8 and q['gap']<=60 and first-w['start']<=60 and w['end']-last<=60
            for j,sensor in enumerate(SENSORS):
                n=q['n'][j];mean=q['sum'][j]/n if n else np.nan;std=np.sqrt(max(0,(q['ss'][j]-q['sum'][j]**2/n)/(n-1))) if n>1 else np.nan
                ma=q['ma_sum'][j]/q['ma_n'][j] if q['ma_n'][j] else np.nan;old=q['old_sum'][j]/q['old_n'][j] if q['old_n'][j] else np.nan
                windowrows.append(dict(**w,sensor=sensor,n=n,mean=mean,std=std,change=q['last'][j]-q['first'][j],trailing_ma15=ma,ma15_change=ma-old,coverage=q['duration']/3600,max_gap=q['gap'],quality=quality,first_time=first,last_time=last))
            for process,b in q['process'].items():
                for j,sensor in enumerate(SENSORS):
                    if b[0][j]>=30:prows.append(dict(Tool=tool,event=w['event'],horizon=w['horizon'],prediction_time=w['end'],stage=process[0],recipe=process[1],recipe_step=process[2],sensor=sensor,n=b[0][j],mean=b[1][j]/b[0][j]))
        inventory.append(dict(Tool=tool,rows=total,temporal_duplicate_rows_dropped=dedup,sample_rows=len(sm),time_recipe_step_strata=len(counts_a),process_strata=len(counts_b),valid_control_hours=len(validhours),low_unique=len(lowtimes),low_cluster_anchors=len(anchors),bytes=file.stat().st_size,mtime_ns=file.stat().st_mtime_ns))
        print(tool,'rows',total,'sample',len(sm),'anchors',len(anchors),'reference hours',len(validhours),flush=True)
        save('equipment_progress',pd.DataFrame(inventory))
    sample=pd.concat(samplelist,ignore_index=True)
    sample=sample.drop(columns=['_a','_b','_priority']).sort_values(['Tool','time','_row'])
    save('profiling_sample',sample);save('window_sensor_statistics',pd.DataFrame(windowrows));save('window_process_statistics',pd.DataFrame(prows));save('reference_process_hours',pd.DataFrame(controls));save('hour_quality',pd.DataFrame(hourly_quality));save('sampling_time_process_strata',pd.DataFrame(strata_a));save('sampling_process_strata',pd.DataFrame(strata_b));save('cluster_sensitivity',pd.DataFrame(cluster_sensitivity))
    print('STREAM COMPLETE',len(sample),'sample rows',flush=True)

def main():
    if '--scan-only' not in sys.argv:prepare()
    scan()
    command([P/'.venv-eda/Scripts/python.exe',P/'src/eda3_reports.py'])
if __name__=='__main__':main()
