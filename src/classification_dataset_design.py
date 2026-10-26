"""Causal feature-window and competing-risk label design audit; no model fitting."""
from pathlib import Path
import sys, argparse, hashlib, json, itertools, html, io, base64, subprocess
from collections import Counter
import numpy as np
import pandas as pd

ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--reuse-features',action='store_true');args=ap.parse_args()
P=args.project.resolve();sys.path.insert(0,str(P))
from src.config import data_root
ROOT=data_root()/'train';L=P/'.local/classification_design';R=P/'reports';F=P/'figures/classification_design'
W=3600;HORIZONS={'6h':21600,'24h':86400,'7d':604800};LEADS={'30m':1800,'1h':3600,'3h':10800};QC={'A_strict':(.8,60),'B_gap300':(.8,300),'C_coverage60':(.6,300)}
SENSORS=pd.read_csv(next(ROOT.glob('[0-9][0-9]_M[0-9][0-9]_DC_train.csv')),nrows=0).columns.tolist()[7:]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,d):d.to_csv(L/(name+'.csv'),index=False)
def snapshot():
    assert not (R/'classification_dataset_design.html').exists(),'Requested new report already exists'
    assert not L.exists(),'Existing design working directory; use a deliberate resume'
    L.mkdir(parents=True);F.mkdir(parents=True,exist_ok=False)
    existing={str(p.relative_to(P)):sha(p) for folder in ['src','reports','figures'] for p in (P/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    (L/'preservation_hashes.json').write_text(json.dumps(existing),encoding='utf-8')
    dest=P/'src/classification_dataset_design.py';assert not dest.exists()
    dest.write_text(Path(__file__).read_text(encoding='utf-8'),encoding='utf-8')

def state():return dict(n=np.zeros(17),sum=np.zeros(17),ss=np.zeros(17),lo=np.full(17,np.inf),hi=np.full(17,-np.inf),first=None,last=None,rows=0,first_t=None,last_t=None,duration=0.,max_gap=0.,process=Counter(),stages=Counter(),recipes=Counter(),steps=Counter(),switches=0,last_process=None)
def accumulate(q,a,t,process,stages,recipes,steps):
    finite=np.isfinite(a);x=np.where(finite,a,0.);q['n']+=finite.sum(0);q['sum']+=x.sum(0);q['ss']+=(x*x).sum(0)
    q['lo']=np.minimum(q['lo'],np.where(finite,a,np.inf).min(0));q['hi']=np.maximum(q['hi'],np.where(finite,a,-np.inf).max(0));q['rows']+=len(t)
    if q['last_t'] is not None:
        gap=float(t[0]-q['last_t']);q['max_gap']=max(q['max_gap'],gap);q['duration']+=min(gap,4.)
    diff=np.diff(t)
    if len(diff):q['max_gap']=max(q['max_gap'],float(diff.max()));q['duration']+=np.minimum(diff,4).sum()
    if q['first_t'] is None:q['first_t']=float(t[0]);q['first']=a[0].copy()
    q['last_t']=float(t[-1]);q['last']=a[-1].copy()
    if q['last_process'] is not None:q['switches']+=int(q['last_process']!=process[0])
    q['switches']+=int(np.count_nonzero(process[1:]!=process[:-1]));q['last_process']=process[-1]
    for key,arr in [('process',process),('stages',stages),('recipes',recipes),('steps',steps)]:q[key].update(arr.tolist())

def scan():
    features=[];source=[];total_duplicates=0
    known=pd.read_csv(R/'sensor_validation/equipment_validation.csv').set_index('Tool')
    for file in sorted(ROOT.glob('[0-9][0-9]_M[0-9][0-9]_DC_train.csv')):
        tool=''.join(file.stem.split('_')[:2]);meta=(file.stat().st_size,file.stat().st_mtime_ns);states={};prev=None;rows=0;duplicates=0
        for d in pd.read_csv(file,chunksize=100000):
            t=d.time.to_numpy(dtype=float);assert np.isfinite(t).all() and np.all(np.diff(t)>=0)
            dup=d.time.duplicated().to_numpy(copy=True)
            if prev is not None:assert t[0]>=prev;dup[0]|=t[0]==prev
            prev=t[-1];rows+=len(d);duplicates+=int(dup.sum());d=d.loc[~dup].copy()
            if d.empty:continue
            t=d.time.to_numpy(dtype=float);a=d[SENSORS].to_numpy(dtype=float);hours=(t//W).astype(np.int64);d['_hour']=hours
            process=pd.util.hash_pandas_object(d[['stage','recipe','recipe_step']],index=False).to_numpy();stage=d.stage.to_numpy();recipe=d.recipe.to_numpy();step=d.recipe_step.to_numpy()
            for hour,idx in d.groupby('_hour',sort=False).indices.items():accumulate(states.setdefault(int(hour),state()),a[idx],t[idx],process[idx],stage[idx],recipe[idx],step[idx])
        assert rows==int(known.loc[tool,'rows']);assert duplicates==int(known.loc[tool,'zero_intervals']);assert meta==(file.stat().st_size,file.stat().st_mtime_ns)
        toolrows=[]
        for hour,q in states.items():
            anchor=(hour+1)*W;r=dict(Tool=tool,hour=hour,anchor=anchor,window_start=hour*W,rows=q['rows'],first_time=q['first_t'],last_time=q['last_t'],coverage=q['duration']/W,max_gap=q['max_gap'],start_age=q['first_t']-hour*W,end_age=anchor-q['last_t'],process_count=len(q['process']),stage_count=len(q['stages']),recipe_count=len(q['recipes']),step_count=len(q['steps']),process_dominant_fraction=max(q['process'].values())/q['rows'],process_switches=q['switches'])
            for field,key in [('stage','stages'),('recipe','recipes'),('recipe_step','steps')]:r[field+'_mode']=q[key].most_common(1)[0][0]
            complete=q['n']/q['rows'];r['min_sensor_completeness']=complete.min();r['critical_completeness']=min(complete[SENSORS.index('FLOWCOOLPRESSURE')],complete[SENSORS.index('FLOWCOOLFLOWRATE')])
            for j,s in enumerate(SENSORS):
                n=q['n'][j];r[s+'__mean']=q['sum'][j]/n if n else np.nan;r[s+'__std']=np.sqrt(max(0.,(q['ss'][j]-q['sum'][j]**2/n)/(n-1))) if n>1 else np.nan;r[s+'__min']=q['lo'][j] if n else np.nan;r[s+'__max']=q['hi'][j] if n else np.nan;r[s+'__change']=q['last'][j]-q['first'][j];r[s+'__valid_n']=int(n);r[s+'__missing_fraction']=1-complete[j]
            toolrows.append(r)
        features.append(pd.DataFrame(toolrows));total_duplicates+=duplicates
        source.append(dict(Tool=tool,raw_rows=rows,dropped_duplicate_times=duplicates,hourly_windows=len(toolrows),bytes=meta[0],mtime_ns=meta[1],unchanged=True))
        save('scan_progress',pd.DataFrame(source));print(tool,rows,'unique-time windows',len(toolrows),flush=True)
    assert total_duplicates==117533
    result=pd.concat(features,ignore_index=True).sort_values(['Tool','anchor']);assert ((result.first_time>=result.window_start)&(result.last_time<result.anchor)).all()
    save('hour_features',result);print('FEATURE SCAN COMPLETE',len(result),flush=True)

def episodes(faults,gap=3600):
    rows=[]
    for tool,g in faults[faults.fault_name.eq('FlowCool Pressure Dropped Below Limit')].groupby('Tool'):
        t=np.sort(g.time.unique());groups=np.cumsum(np.r_[True,np.diff(t)>gap])
        for group in np.unique(groups):
            a=t[groups==group];rows.append(dict(Tool=tool,event_start=int(a[0]),event_end=int(a[-1]),raw_unique_records=len(a),episode_id=tool+':'+str(int(a[0]))))
    return pd.DataFrame(rows,columns=['Tool','event_start','event_end','raw_unique_records','episode_id']).sort_values(['Tool','event_start'])

def horizon_followup(tool_features,anchors,horizon,origin,finish):
    # Future coverage is a LABEL availability test only, never a predictor.
    start=int(origin//W);end=int(finish//W)+2;present=np.zeros(end-start,dtype=np.int8)
    present[(tool_features.hour.astype(int)-start).to_numpy()]=1
    n=horizon//W;offset=anchors//W-start;cs=np.r_[0,np.cumsum(present)];right=np.minimum(offset+n,len(present));left=np.minimum(offset,len(present));frac=(cs[right]-cs[left])/n
    # Independent direct sliding availability calculation; no interpolation.
    max_empty=np.zeros(len(anchors),dtype=int);run=np.zeros(len(anchors),dtype=int)
    for j in range(n):
        ix=offset+j;observed=np.zeros(len(anchors),dtype=bool);ok=(ix>=0)&(ix<len(present));observed[ok]=present[ix[ok]].astype(bool);run=np.where(observed,0,run+1);max_empty=np.maximum(max_empty,run)
    return frac,max_empty

def classify(a,event_starts,event_ends,event_ids,other,all_times,last_time,horizon,lead,cooldown,followup_ok):
    # Mutually exclusive exclusions in a documented fixed order.
    n=len(a);status=np.full(n,'eligible',dtype=object);event=np.full(n,'',dtype=object);next_low=np.full(n,np.inf);next_other=np.full(n,np.inf);last_fault=np.full(n,-np.inf)
    ix=np.searchsorted(event_starts,a,side='right');ok=ix<len(event_starts);next_low[ok]=event_starts[ix[ok]];event[ok]=event_ids[ix[ok]]
    # Any low episode still in progress is an already-known condition, not a future target.
    back=np.searchsorted(event_starts,a,side='right')-1;ongoing=np.zeros(n,dtype=bool);ok=back>=0;ongoing[ok]=event_ends[back[ok]]>=a[ok]
    ix=np.searchsorted(other,a,side='right');ok=ix<len(other);next_other[ok]=other[ix[ok]]
    ix=np.searchsorted(all_times,a,side='right')-1;ok=ix>=0;last_fault[ok]=all_times[ix[ok]]
    status[ongoing | ((a-last_fault)<=cooldown)]='post_fault_or_ongoing'
    remaining=status=='eligible';status[remaining & ((a+horizon)>last_time)]='right_censored_end'
    remaining=status=='eligible';status[remaining & (next_low<=a+lead)]='imminent_blind_zone'
    positive=(next_low>a+lead)&(next_low<=a+horizon)
    remaining=status=='eligible';competing=next_other<=np.where(positive,next_low,a+horizon);status[remaining & competing]='competing_fault'
    remaining=status=='eligible';status[remaining & ~positive & ~followup_ok]='uncertain_negative_followup'
    y=np.full(n,-1,dtype=int);good=status=='eligible';y[good]=positive[good].astype(int);event[y!=1]=''
    return y,status,event,next_low

def independent_counts(g,horizon):
    # One risk row per episode, then disjoint complete information supports.
    pos=g[g.y.eq(1)].sort_values('anchor').drop_duplicates('episode_id');selected=[];last=-np.inf
    for r in pos.itertuples():
        if r.anchor-W>=last:selected.append((r.anchor-W,r.anchor+horizon));last=r.anchor+horizon
    neg=g[g.y.eq(0)].sort_values('anchor');count=0;last=-np.inf
    for r in neg.itertuples():
        left=r.anchor-W;right=r.anchor+horizon
        if left<last or any(left<b and right>a for a,b in selected):continue
        count+=1;last=right
    return len(pos),len(selected),count

def synthetic_tests():
    # Label boundaries, competing fault ordering, censoring, and post-fault exclusions.
    t=np.array([10000]);empty=np.array([],dtype=float)
    def label(f,lead=1000,other=empty,last=50000,prior=empty):return classify(t,np.array([f]),np.array([f]),np.array(['E']),np.asarray(other),np.asarray(prior),last,10000,lead,0,np.ones(1,dtype=bool))
    assert label(11000)[1][0]=='imminent_blind_zone'
    assert label(20000)[0][0]==1 and label(20001)[0][0]==0
    assert label(15000,other=[14000])[1][0]=='competing_fault'
    assert label(15000,other=[16000])[0][0]==1
    assert label(15000,last=19000)[1][0]=='right_censored_end'
    assert label(15000,prior=[10000])[1][0]=='post_fault_or_ongoing'
    a=np.arange(68,dtype=float).reshape(4,17);a[1,2]=np.nan;q=state();tt=np.array([0.,4.,8.,100.]);p=np.ones(4)
    accumulate(q,a[:2],tt[:2],p[:2],p[:2],p[:2],p[:2]);accumulate(q,a[2:],tt[2:],p[2:],p[2:],p[2:],p[2:]);np.testing.assert_allclose(q['sum']/q['n'],np.nanmean(a,axis=0));assert q['max_gap']==92 and q['duration']==12

def design():
    synthetic_tests();features=pd.read_csv(L/'hour_features.csv');faults=pd.concat([pd.read_csv(p) for p in ROOT.joinpath('train_faults').glob('*_train_fault_data.csv')],ignore_index=True).drop_duplicates(['Tool','time','fault_name']);ep=episodes(faults);save('low_pressure_episodes',ep)
    boundaries=pd.read_csv(R/'sensor_validation/equipment_validation.csv').set_index('Tool')
    counts=[];exclusions=[];splits=[];selected_rows=[];sensitivity=[];feature_schema=[]
    for name in features.columns:
        role='feature' if '__' in name or name in ['coverage','max_gap','start_age','end_age','process_count','stage_count','recipe_count','step_count','process_dominant_fraction','process_switches','stage_mode','recipe_mode','recipe_step_mode'] else 'audit_only'
        feature_schema.append(dict(column=name,role=role,source='past_window_only',preprocessing='Fit any imputation/encoding/scaling on training split only' if role=='feature' else 'Excluded from model inputs'))
    save('feature_schema',pd.DataFrame(feature_schema))
    for tool,g in features.groupby('Tool',sort=True):
        g=g.sort_values('anchor').reset_index(drop=True);a=g.anchor.astype(int).to_numpy();b=boundaries.loc[tool];origin=float(b.first_time);finish=float(b.last_time);events=ep[ep.Tool.eq(tool)];es=events.event_start.to_numpy();ee=events.event_end.to_numpy();eid=events.episode_id.to_numpy();fg=faults[faults.Tool.eq(tool)];other=np.sort(fg.loc[fg.fault_name.ne('FlowCool Pressure Dropped Below Limit'),'time'].unique());alltimes=np.sort(fg.time.unique())
        quality={name:(g.coverage.ge(cov)&g.max_gap.le(gap)&g.start_age.le(gap)&g.end_age.le(gap)&g.rows.ge(100)&g.critical_completeness.ge(.95)&g.min_sensor_completeness.ge(.9)).to_numpy() for name,(cov,gap) in QC.items()}
        for hlabel,h in HORIZONS.items():
            followfrac,emptyrun=horizon_followup(g,a,h,origin,finish)
            for llabel,lead in LEADS.items():
                for recovery in [21600,86400]:
                    for followname,follow in [('endpoint_only',np.ones(len(g),dtype=bool)),('observed50_no_gap6h',(followfrac>=.5)&(emptyrun<=6))]:
                        y,status,events_id,next_low=classify(a,es,ee,eid,other,alltimes,finish,h,lead,recovery,follow)
                        for qname,qmask in quality.items():
                            config=dict(Tool=tool,horizon=hlabel,min_lead=llabel,quality=qname,post_fault_hours=recovery//3600,followup=followname)
                            valid=qmask & (y>=0);z=pd.DataFrame(dict(anchor=a[valid],y=y[valid],episode_id=events_id[valid]));event_n,isolated_pos,isolated_neg=independent_counts(z,h)
                            n0=int(np.sum(valid & (y==0)));n1=int(np.sum(valid & (y==1)))
                            counts.append(dict(**config,candidate_windows=len(g),quality_windows=int(qmask.sum()),negative=n0,positive=n1,total=n0+n1,positive_fraction=n1/(n0+n1) if n0+n1 else np.nan,unique_positive_episodes=event_n,disjoint_positive_blocks=isolated_pos,disjoint_negative_blocks=isolated_neg))
                            st=np.where(~qmask,'past_window_quality',status)
                            for cause,n in Counter(st).items():exclusions.append(dict(**config,status=cause,count=n))
                            if qname=='B_gap300' and llabel=='1h' and recovery==21600:
                                # Reference design for auditing only; no final setting/model selection.
                                cut1=origin+.6*(finish-origin);cut2=origin+.8*(finish-origin);start=a-W;end=a+h;fold=np.full(len(g),'purged_boundary',dtype=object)
                                fold[(start>=origin)&(end<=cut1)]='train';fold[(start>=cut1)&(end<=cut2)]='validation';fold[(start>=cut2)&(end<=finish)]='test'
                                for split in ['train','validation','test','purged_boundary']:
                                    mask=valid&(fold==split);splits.append(dict(**config,split=split,negative=int(np.sum(mask&(y==0))),positive=int(np.sum(mask&(y==1))),unique_positive_episodes=len(set(events_id[mask & (y==1)]))))
                                out=g.loc[valid,['Tool','anchor','window_start','last_time']].copy();out['horizon']=hlabel;out['followup']=followname;out['label']=y[valid];out['episode_id']=events_id[valid];out['target_time']=np.where(y[valid]==1,next_low[valid],np.nan);out['split']=fold[valid];out['label_end']=end[valid];selected_rows.append(out)
                                # Verify one event cannot occur in different retained time folds.
                                retained=out[out.label.eq(1)&out.split.ne('purged_boundary')];assert retained.groupby('episode_id').split.nunique().le(1).all()
            # Episode-definition sensitivity under the B/1h/6h/endpoint reference.
            for gap in [3600,21600,86400]:
                e=episodes(fg,gap);y,status,eids,_=classify(a,e.event_start.to_numpy(),e.event_end.to_numpy(),e.episode_id.to_numpy(),other,alltimes,finish,h,3600,21600,np.ones(len(a),dtype=bool));valid=quality['B_gap300']&(y>=0)
                sensitivity.append(dict(Tool=tool,horizon=hlabel,cluster_gap_hours=gap//3600,episode_count=len(e),positive=int(np.sum(valid&(y==1))),negative=int(np.sum(valid&(y==0))),represented_positive_episodes=len(set(eids[valid&(y==1)]))))
        print('LABELS',tool,flush=True)
    c=pd.DataFrame(counts);save('counts_by_equipment',c);save('exclusions_by_equipment',pd.DataFrame(exclusions));save('split_counts',pd.DataFrame(splits));save('reference_sample_index',pd.concat(selected_rows,ignore_index=True));save('episode_sensitivity',pd.DataFrame(sensitivity))
    group=['horizon','min_lead','quality','post_fault_hours','followup'];agg=c.groupby(group,as_index=False)[['candidate_windows','quality_windows','negative','positive','total','unique_positive_episodes','disjoint_positive_blocks','disjoint_negative_blocks']].sum();agg['positive_fraction']=agg.positive/agg.total.replace(0,np.nan);save('counts_all_configurations',agg)
    # Exact counts reconcile with mutually exclusive exclusions, not overlapping reason totals.
    ex=pd.DataFrame(exclusions);check=ex.groupby(['Tool']+group)['count'].sum().reset_index();assert np.array_equal(check['count'].to_numpy(),c.sort_values(['Tool']+group).candidate_windows.to_numpy())
    assert ((features.first_time>=features.window_start)&(features.last_time<features.anchor)).all()
    assert len(c)==20*3*3*3*2*2
    build_report(features,ep,c,agg,pd.DataFrame(splits),pd.DataFrame(sensitivity),ex)

def build_report(features,ep,c,agg,splits,sensitivity,ex):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']=['Malgun Gothic','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    def table(d):return d.to_html(index=False,border=0,float_format=lambda x:f'{x:,.4g}',na_rep='해당 없음')
    def img(fig,name):
        fig.tight_layout();fig.savefig(F/(name+'.svg'));buf=io.BytesIO();fig.savefig(buf,format='png',dpi=140);plt.close(fig);return '<img alt="'+name+'" src="data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()+'">'
    reference=agg[(agg.quality=='B_gap300')&(agg.min_lead=='1h')&(agg.post_fault_hours==6)]
    parts=['<h1>PHM 2018 이진 분류 학습 데이터 설계 및 표본 수 검증</h1><p>목표: 미래 냉각 압력 저하 고장 기록 발생 여부. 모델 학습, 성능 산출, 최종 예측 기간·모델 선정은 수행하지 않았다.</p><p><b>time=초라는 작업 가정</b> 아래 계산했다. 공식 설명은 고장 time을 “예: 초”라고 표기하며 확정적인 단위·기준 날짜를 제공하지 않는다. 시간 한 단위가 실제 κ초이면 아래 H시간과 과거 1시간은 각각 실제 κH시간과 κ시간이다. 밀리초라면 6시간 설정은 실제 21.6초, 분이라면 실제 360시간(15일)이 된다. 숫자 경계를 그대로 둘 때 행 수 자체는 같아도 예측 문제의 물리적 의미가 달라진다. 올바른 시간 단위가 확정되면 원래 의도한 6시간·24시간·7일 경계로 다시 집계해야 한다.</p>']
    parts+=['<h2>입력·시간창·라벨 정의</h2><p>장비별 원본 time이 같은 측정은 청크 경계를 포함해 첫 행만 유지했다. 117,533개 반복 시각 행을 제거했다. 다른 값의 동시 측정 가능성은 별도 미해결 사항이다. 1시간 고정 격자 t에서 과거 [t−1시간,t)만 읽어 특징을 계산했다. 특징 창은 같은 장비 안에서 겹치지 않는다. 격자 끝 시각의 센서는 다음 창에 속하며 현재 창 특징에는 들어가지 않는다. 고장과 TTF는 특징에 포함하지 않는다.</p><p>H는 예측 시점에서 가장 먼 미래 경계(6시간·24시간·7일), L은 최소 선행시간(30분·1시간·3시간)이다. 양성은 다음 저압 사건 시작 F가 (t+L,t+H]에 있는 경우다. 따라서 실제 양성 라벨 구간 길이는 H−L이며, “최소 선행시간 이후에 H시간 길이”라는 다른 정의와 구분한다. (t,t+L]에 저압 사건이 있으면 임박 고장으로 제외하고 음성으로 처리하지 않는다. 0은 해당 구간에 기록된 저압 사건이 없다는 뜻이며 물리적 정상 보증이 아니다.</p><p>동일 (Tool,time,fault_name)을 제거한 저압 기록은 741개다. 같은 장비에서 인접 저압 간격 ≤1시간을 연결 군집으로 묶고 최초 시각을 목표 사건 시작으로 정했다. 마지막 기록까지는 진행 중 사건으로 처리한다. 총 '+str(len(ep))+'개 운영상 사건 군집이며, 수리 완료와 기계적 독립성을 입증한 개수는 아니다. 6시간·24시간 병합 민감도도 아래에 제시했다.</p>']
    parts+=['<h2>특징 사전과 공정 조건</h2><p>센서 17개 각각 평균(sum/n), 표준편차(ddof=1), 최솟값, 최댓값, 변화량(마지막−첫 값)을 계산했다. 유효값 수와 결측 비율도 기록했다. 변화량의 첫/끝 값이 결측이면 변화량도 결측으로 유지한다. 미래 보간·양방향 보간·중앙 이동창을 쓰지 않았다.</p><p>과거 창의 stage·recipe·recipe_step 최빈값, 고유 공정 수, 주 공정 비중, 공정 전환 수를 함께 기록했다. 정확히 같은 공정의 정상 비교 구간을 반드시 찾는 EDA 조건은 분류 표본 요건으로 사용하지 않았다. 서로 다른 공정 혼합을 무시한 것이 아니라, 과거에 관측 가능한 공정 구성과 누락·공백 지표를 검증/특징 후보로 남겼다. Lot·runnum·Tool·절대 time과 라벨 감사 열은 기본 모델 입력에서 제외하는 설계다. 공정 인코딩·미지 범주 처리·결측 대체·정규화·상수/변수 선택은 학습 분할만 사용해야 하며 이번에는 수행하지 않았다.</p><p>특징 후보와 감사 열을 분리한 feature_schema.csv 및 실제 과거 특징 hour_features.csv를 Git 제외된 .local/classification_design에 저장했다. 표본 수 비교는 실제 특징 창의 관측 품질을 바탕으로 계산한 값이다.</p>']
    parts+=['<h2>배제·검열 기준</h2><p>배제는 고정된 우선순위로 한 표본에 한 사유를 부여했다: ① 과거 창 품질 실패, ② 진행 중 저압 사건 또는 모든 유형의 직전 고장 후 6시간/24시간, ③ t+H가 장비의 마지막 관측 이후인 우측 검열, ④ 최소 선행시간 안 임박 저압 사건, ⑤ 목표 저압보다 먼저 또는 동시에 다른 유형 고장 발생(양성), 혹은 t+H 이내 다른 고장 발생(음성 후보), ⑥ 음성 후속 관측 부족. 저압 사건 이후 발생한 다른 고장은 이미 관측된 양성 라벨을 취소하지 않는다.</p><p>직전 고장 후 대기 시간은 수리 완료를 모르는 상태의 보수적 대체 기준이며 정비 복구가 검증됐다는 뜻이 아니다. 첫/마지막 사건 사이 고장 로그의 완전성도 확인되지 않았다. t+H까지 전부 관측되는 창만 사용해 양성·음성의 종료 검열 규칙을 일관되게 적용했다. 그래서 마지막 기록 직전의 알려진 고장도 완전한 예측 기간 요건 때문에 제외될 수 있다.</p><p>endpoint_only는 파일 마지막 시각까지의 고장 로그 완전성을 가정한다. observed50_no_gap6h는 음성 후보의 미래 1시간 bin 중 실제 센서가 있는 비율 ≥50%, 연속 센서 없는 시간 ≤6시간까지 추가 요구한다. 이는 라벨 관측 가능성 검사이고 미래 센서값·관측률을 특징으로 사용하지 않는다. 센서가 드문 공정은 음성에서 더 많이 제외될 수 있어 두 설정의 비율을 함께 보고한다. 양성은 고장 기록으로 확인되므로 후속 센서 가용성 조건은 음성에만 적용한다. 이것이 유발하는 선택 편향도 모델 성능 평가 전에 검토해야 한다.</p>']
    parts+=['<h2>관측 품질 후보</h2><p>A: 관측 비율 ≥80%, 최대 공백·양 끝 미관측 각각 ≤60초. B: ≥80%, ≤300초. C: ≥60%, ≤300초. 모든 설정에서 고유 시각 ≥100개, 압력·유량 유효 비율 ≥95%, 모든 센서 유효 비율 ≥90%를 요구한다. 관측 비율은 양의 인접 간격을 최대 4초로 제한해 합한 값을 3,600으로 나눈 값이다. 4초는 앞선 전수 검사의 측정 간격 중앙값이다. 긴 공백을 관측시간으로 채우지 않는다. C는 더 불완전한 창을 허용하므로 자동 채택하지 않는다. 과거 공정 일치 조건을 제거했다고 해서 공백·결측을 채워 수량을 늘린 것은 아니다.</p>']
    parts+=['<h2>기준 비교 결과</h2><p>아래 B·최소 선행 1시간·고장 후 6시간 기준은 집계와 분할 설명을 위한 참조 설정이며 최종 선택이 아니다. 두 라벨 관측 기준을 나란히 비교한다.</p>',table(reference)]
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,follow in zip(axes,['endpoint_only','observed50_no_gap6h']):
        g=reference[reference.followup.eq(follow)].set_index('horizon').reindex(HORIZONS)
        ax.bar(g.index,g.negative,label='0: 기록된 저압 없음');ax.bar(g.index,g.positive,bottom=g.negative,label='1: 미래 저압');ax.set(title=follow,ylabel='비중첩 과거 창 수');ax.legend(fontsize=8)
    parts.append(img(fig,'class_counts'))
    parts+=['<h2>예측 기간·최소 선행·품질·복구·검열 전체 비교</h2><p>총 108개 설정(3×3×3×2×2)을 비교했다. positive_fraction은 1/(0+1)이며, 배제된 창은 분모에 넣지 않는다.</p>',table(agg)]
    device=c[(c.quality=='B_gap300')&(c.min_lead=='1h')&(c.post_fault_hours==6)]
    parts+=['<h2>장비별 클래스 수·비율·사건 수</h2>',table(device)]
    parts+=['<h2>독립 사건·비고장 비교 단위</h2><p>unique_positive_episodes는 양성 창에서 실제로 대표된 고유 사건 군집 수다. positive 행 수와 같지 않다. 각 사건의 가장 이른 적격 예측 창 하나만 남긴 뒤 [t−1시간,t+H] 전체 정보 구간이 겹치지 않도록 탐욕적으로 선택한 개수를 disjoint_positive_blocks로 보고했다. 이후 그 양성 정보 구간과 겹치지 않고, 음성끼리도 겹치지 않는 구간을 시간순으로 선택한 개수가 disjoint_negative_blocks다. 기계적 독립성 증명이 아닌, 시간 정보 중첩을 제거한 비교 단위의 보수적 개수다. 다른 선택 순서에서는 수가 달라질 수 있다.</p><p>실제 모델 설계에서는 모든 적격 과거 창을 유지하되 같은 사건 양성 창의 가중치 합을 1로 맞추거나 사건 단위 샘플링을 검토할 수 있다. 어느 방법도 이번에 확정하지 않았다. 이벤트 라벨/검열/선택 정보를 모델 입력에 넣지 않으며, 균형화·리샘플링은 학습 분할 안에서만 수행해야 한다.</p>',table(reference[['horizon','followup','positive','negative','unique_positive_episodes','disjoint_positive_blocks','disjoint_negative_blocks']])]
    sens=sensitivity.groupby(['horizon','cluster_gap_hours'],as_index=False)[['episode_count','positive','negative','represented_positive_episodes']].sum()
    parts+=['<h3>반복 사건 정의 민감도</h3><p>B·선행 1시간·복구 6시간·endpoint_only에서 사건 병합 간격만 변경했다. 모든 시간 설정의 전 조합을 재실행한 것은 아니다.</p>',table(sens)]
    splitagg=splits.groupby(['horizon','followup','split'],as_index=False)[['negative','positive','unique_positive_episodes']].sum()
    parts+=['<h2>시간순 학습·검증·테스트 설계</h2><p>각 장비의 첫~마지막 관측 시간축을 60%/20%/20%로 나눴다. [t−1시간,t+H] 전체가 같은 구간 안에 있는 표본만 해당 분할에 남겼다. 경계를 가로지르면 purged_boundary로 제외했다. 경계 직전 H시간과 직후 1시간의 정보 중첩이 자동 제거된다. 분할별 결과는 참조 B·선행1h·복구6h 설정이다. 같은 목표 사건이 여러 유지 분할에 걸치지 않는지 검사했다.</p><p>사건 군집의 전체 [최초,마지막] 기록 구간이 분할 경계를 넘는 경우에는 해당 군집 전체를 두 인접 분할에서 제외하는 추가 규칙을 권고한다. 아래 현재 집계는 목표 최초 시각과 전체 표본 정보 구간의 격리를 구현했으며, 군집 전체 경계 배제를 추가 적용한 최종 학습 데이터는 아직 확정하지 않았다. 7일 예측은 purge와 검열 때문에 검증·테스트 사건이 매우 적을 수 있다.</p>',table(splitagg),'<details><summary>장비별 분할 상세</summary>',table(splits),'</details>']
    parts+=['<h2>장비 일반화 검증 방안</h2><p>시간순 검증과 별도로, 장비 하나를 완전히 제외하는 leave-one-tool-out 외부 검증 20회 또는 사전 지정 장비 그룹 분리를 제안한다. 외부 장비의 센서·분포·라벨로 전처리나 특성 선택을 맞추지 않는다. 나머지 장비 내부에서도 시간순 학습/검증을 유지하고, 미관측 장비 평가는 미리 정한 후기 구간에 한정한다. 장비별 절대 시계가 같은 달력 기준인지 미검증이므로, 장비별 상대 시간 분할이 실제 동시점 배포의 전향 평가와 같다고 주장하지 않는다.</p><p>장비별 표에서 양성 0인 장비도 삭제하지 않고 비고장/거짓 경보 검증 대상으로 남긴다. 해당 장비 단독 재현율·ROC-AUC 등 양성 의존 지표는 정의되지 않을 수 있다. 모델 개발 시 사건 단위 탐지율, 선행시간, 운용시간당 거짓 경보, 장비별 성능과 PR 기반 지표를 함께 검토해야 하며 이번에 계산하지 않았다. 장비의 선택은 전체 데이터의 성능을 본 뒤 정하지 않아야 한다.</p>']
    exagg=ex.groupby(['horizon','min_lead','quality','post_fault_hours','followup','status'],as_index=False)['count'].sum()
    exref=exagg[(exagg.quality=='B_gap300')&(exagg.min_lead=='1h')&(exagg.post_fault_hours==6)]
    parts+=['<h2>제외 사유 확인</h2><p>서로 겹치는 사유의 단순 합이 아니다. 앞에서 정의한 우선순위로 분류하여 각 설정의 후보 창 수와 정확히 합계가 일치함을 확인했다.</p>',table(exref)]
    old=json.loads((L/'preservation_hashes.json').read_text());assert all(sha(P/rel)==h for rel,h in old.items())
    parts+=['<h2>검증 범위·미확정 사항</h2><p>경계 단위 검사(최소 선행/예측 종료, 경쟁 고장 순서, 관측 종료, 고장 직후), 청크 경계 결측 평균·공백 집계, 반복 시각 제거 수, 전체 배제 합계와 시간창 인과성을 검사했다. 원본 파일은 읽기 전후 크기·수정시각이 같았고, 기존 코드·보고서·그래프 '+str(len(old))+'개는 SHA-256이 일치했다. 모델 학습·스케일링·대체·인코더 학습은 하지 않았다.</p><p>시간 단위·고장 로그 완전성·정비 완료·군집의 기계적 독립성·미래 센서 기록 부재의 원인·공정 간 교란은 미검증이다. 표본 수가 많다는 이유만으로 B 또는 C를 확정하지 않았다. 이 보고서는 실제 특징 창과 라벨 후보 수를 검증한 설계 결과이며, 최종 학습 설정과 모델 성능 결과는 아니다.</p>']
    for name in ['counts_all_configurations','counts_by_equipment','split_counts','episode_sensitivity','feature_schema']:
        parts.append('<p>세부 집계: .local/classification_design/'+name+'.csv (Git 제외)</p>')
    parts.append('<p><a href="fault_analysis/report.html">고장 기록</a> · <a href="sensor_validation/sensor_validation_report.html">센서 검증</a> · <a href="sensor_profiling_report.html">YData</a> · <a href="timeseries_eda_report.html">시계열 EDA</a></p>')
    style='body{font-family:Malgun Gothic,sans-serif;max-width:1500px;margin:32px auto;padding:0 22px;color:#172d42;line-height:1.7}table{display:block;overflow:auto;border-collapse:collapse;font-size:12px}th,td{padding:7px 10px;border-bottom:1px solid #ddd;white-space:nowrap;text-align:right}th{background:#eaf1f8}img{width:100%}h2{margin-top:42px}'
    (R/'classification_dataset_design.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>PHM2018 이진 분류 데이터 설계</title><style>'+style+'</style>'+''.join(parts)+'</html>',encoding='utf-8')
    summary=dict(raw_sensor_rows=82189440,hourly_feature_windows=len(features),features_columns=len(features.columns),low_unique_records=741,low_episode_clusters=len(ep),configurations=len(agg),equipment_configuration_rows=len(c),old_files_unchanged=len(old),source_and_boundary_tests='PASS',model_training=False)
    (L/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print('REFERENCE\n'+reference.to_string(index=False));print('SPLITS\n'+splitagg.to_string(index=False));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    if not args.reuse_features:snapshot();scan()
    design()
