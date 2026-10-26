"""First fixed-parameter model comparison. No hyperparameter search or final selection."""
from pathlib import Path
import os
for var in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[var]='4'
import argparse,sys,json,hashlib,subprocess,shutil,time,warnings,io,base64
from collections import Counter
ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--resume',action='store_true');ap.add_argument('--report-only',action='store_true');args=ap.parse_args()
P=args.project.resolve();L=P/'.local/model_comparison';R=P/'reports';F=P/'figures/model_comparison';M=P/'models/model_comparison';SEED=2018;H=86400;W=3600

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare():
    assert not (R/'model_comparison_report.html').exists() and not L.exists()
    L.mkdir(parents=True);F.mkdir(parents=True,exist_ok=False);M.mkdir(parents=True,exist_ok=False)
    old={str(p.relative_to(P)):sha(p) for folder in ['src','reports','figures','models'] for p in (P/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    (L/'preservation_hashes.json').write_text(json.dumps(old),encoding='utf-8')
    dest=P/'src/model_comparison.py';assert not dest.exists();dest.write_text(Path(__file__).read_text(encoding='utf-8'),encoding='utf-8')
    req=P/'requirements-model.txt';assert not req.exists();req.write_text('scikit-learn==1.9.1\nnumpy==2.3.5\npandas==3.0.1\nmatplotlib==3.10.0\n',encoding='utf-8')
    venv=P/'.venv-model';assert not venv.exists();subprocess.run([sys.executable,'-m','venv',str(venv)],check=True)
    py=venv/'Scripts/python.exe';subprocess.run([str(py),'-m','pip','install','--disable-pip-version-check','-r',str(req)],check=True)
    subprocess.run([str(py),'-m','pip','check'],check=True)
    (P/'requirements-model-lock.txt').write_text(subprocess.check_output([str(py),'-m','pip','freeze'],text=True),encoding='utf-8')
    ignore=P/'.gitignore';txt=ignore.read_text(encoding='utf-8');ignore.write_text(txt+'\n# Isolated first model comparison runtime\n.venv-model/\n' if '.venv-model/' not in txt else txt,encoding='utf-8')

def experiment():
    import numpy as np
    import pandas as pd
    import sklearn,joblib
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder,StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
    from sklearn.metrics import precision_recall_curve,average_precision_score,roc_auc_score,confusion_matrix,precision_score,recall_score,f1_score,accuracy_score,brier_score_loss,log_loss
    from sklearn.inspection import permutation_importance
    from threadpoolctl import threadpool_limits
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']=['Malgun Gothic','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    sys.path.insert(0,str(P));from src.config import data_root
    D=P/'.local/classification_design';idx=pd.read_csv(D/'reference_sample_index.csv');idx=idx[idx.horizon.eq('24h')&idx.split.ne('purged_boundary')].copy()
    schema=pd.read_csv(D/'feature_schema.csv');sensors=[c for c in schema.column if any(c.endswith('__'+s) for s in ['mean','std','min','max','change'])]
    process_num=['process_count','stage_count','recipe_count','step_count','process_dominant_fraction','process_switches'];cats=['stage_mode','recipe_mode','recipe_step_mode'];columns=sensors+process_num+cats
    assert len(sensors)==85 and len(columns)==94
    assert set(columns)<=set(schema.loc[schema.role.eq('feature'),'column'])
    assert not any(any(x in c.lower() for x in ['ttf','fault','target','episode','label','tool','anchor','time']) for c in columns if c not in ['ETCHAUXSOURCETIMER__mean','ETCHAUXSOURCETIMER__std','ETCHAUXSOURCETIMER__min','ETCHAUXSOURCETIMER__max','ETCHAUXSOURCETIMER__change','ETCHAUX2SOURCETIMER__mean','ETCHAUX2SOURCETIMER__std','ETCHAUX2SOURCETIMER__min','ETCHAUX2SOURCETIMER__max','ETCHAUX2SOURCETIMER__change'])
    keys=idx[['Tool','anchor']].drop_duplicates();chunks=[]
    for chunk in pd.read_csv(D/'hour_features.csv',usecols=['Tool','anchor']+columns,chunksize=50000):chunks.append(chunk.merge(keys,on=['Tool','anchor'],how='inner',validate='one_to_one'))
    features=pd.concat(chunks,ignore_index=True);df=idx.merge(features,on=['Tool','anchor'],how='left',validate='many_to_one')
    assert len(df)==len(idx)
    for c in cats:df[c]=df[c].map(lambda x:'__MISSING__' if pd.isna(x) else str(x)).astype(object)
    ep=pd.read_csv(D/'low_pressure_episodes.csv');bounds=pd.read_csv(R/'sensor_validation/equipment_validation.csv').set_index('Tool')
    faults=pd.concat([pd.read_csv(p) for p in (data_root()/'train/train_faults').glob('*_train_fault_data.csv')],ignore_index=True).drop_duplicates(['Tool','time','fault_name'])
    # Guard the ENTIRE feature window against the 6h post-fault exclusion, not just its endpoint.
    recovery_ok=np.ones(len(df),dtype=bool);bad_episodes=set()
    for tool,ii in df.groupby('Tool',sort=False).indices.items():
        ft=np.sort(faults.loc[faults.Tool.eq(tool),'time'].unique());a=df.iloc[ii].anchor.to_numpy();j=np.searchsorted(ft,a,side='right')-1;prior=np.full(len(a),-np.inf);ok=j>=0;prior[ok]=ft[j[ok]]
        recovery_ok[ii]=df.iloc[ii].window_start.to_numpy()-prior>21600
        b=bounds.loc[tool];e=ep[ep.Tool.eq(tool)]
        for fraction in [.6,.8]:
            cut=b.first_time+fraction*(b.last_time-b.first_time);bad_episodes.update(e.loc[(e.event_start<=cut)&(e.event_end>=cut),'episode_id'])
    removed=[]
    for follow,g in df.groupby('followup'):
        ii=g.index.to_numpy();removed.append(dict(followup=follow,reference_retained_rows=len(g),feature_recovery_exclusions=int((~recovery_ok[ii]).sum()),boundary_episode_exclusions=int((recovery_ok[ii]&g.episode_id.isin(bad_episodes).to_numpy()).sum())))
    df=df[recovery_ok & ~df.episode_id.isin(bad_episodes)].copy();df['y']=df.label.astype(int)
    for tool,g in df.groupby('Tool'):
        b=bounds.loc[tool];c1=b.first_time+.6*(b.last_time-b.first_time);c2=b.first_time+.8*(b.last_time-b.first_time)
        for fold,lo,hi in [('train',b.first_time,c1),('validation',c1,c2),('test',c2,b.last_time)]:
            q=g[g.split.eq(fold)];assert (q.window_start>=lo).all() and (q.label_end<=hi).all() and (q.last_time<q.anchor).all()
        q=g[g.y.eq(1)];assert q.groupby(['followup','episode_id']).split.nunique().le(1).all()
    endpoint=df[df.followup.eq('endpoint_only')].copy();sensitivity=df[df.followup.eq('observed50_no_gap6h')].copy()
    chk=endpoint[['Tool','anchor','split','y']].merge(sensitivity[['Tool','anchor','split','y']],on=['Tool','anchor'],suffixes=('_e','_s'),validate='one_to_one');assert len(chk)==len(sensitivity) and chk.y_e.eq(chk.y_s).all() and chk.split_e.eq(chk.split_s).all()
    held_tools=sorted(df.Tool.unique())[::5];assert len(held_tools)==4
    metrics=[];event_metrics=[];leads=[];device_metrics=[];process_metrics=[];calibration=[];splits=[];parameters=[];predictions=[];importance=[];warnings_log=[];model_cache={};thresholds=[];prob_cache={};scenario_frames={};weight_records=[]
    def metric(y,p,th):
        y=np.asarray(y,dtype=int);p=np.clip(np.asarray(p,dtype=float),0,1);yp=p>=th;tn,fp,fn,tp=confusion_matrix(y,yp,labels=[0,1]).ravel()
        return dict(n=len(y),positive=int(y.sum()),prevalence=float(y.mean()) if len(y) else np.nan,precision=precision_score(y,yp,zero_division=0),recall=recall_score(y,yp,zero_division=0),f1=f1_score(y,yp,zero_division=0),average_precision=average_precision_score(y,p) if y.sum()>0 else np.nan,roc_auc=roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan,accuracy=accuracy_score(y,yp),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),false_positive_rate=fp/(fp+tn) if fp+tn else np.nan,brier=brier_score_loss(y,p),log_loss=log_loss(y,np.clip(p,1e-12,1-1e-12),labels=[0,1]))
    def threshold(y,p,constant_normal=False):
        if constant_normal or np.sum(y)==0:return .5
        precision,recall,values=precision_recall_curve(y,p);f=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-15)
        best=np.flatnonzero(np.isclose(f,f.max(),atol=1e-12,rtol=0));return float(values[best[-1]]) if len(values) else .5
    def weights(train):
        y=train.y.to_numpy();w=np.ones(len(train));pos=y==1
        n=train.loc[pos].groupby('episode_id').anchor.transform('size').to_numpy();w[pos]=1/n
        for label in [0,1]:mask=y==label;assert mask.any();w[mask]*=len(y)/2/w[mask].sum()
        assert np.isclose(w[y==0].sum(),w[y==1].sum())
        return w
    def pipeline(name,cols):
        cc=[c for c in cats if c in cols];nn=[c for c in cols if c not in cc]
        num=[('impute',SimpleImputer(strategy='median',keep_empty_features=True))]
        if name.startswith('Logistic'):num.append(('scale',StandardScaler()))
        transformers=[('num',Pipeline(num),nn)] if nn else []
        if cc:transformers.append(('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float64),cc))
        pre=ColumnTransformer(transformers,remainder='drop',sparse_threshold=0)
        if name.startswith('Logistic'):est=LogisticRegression(random_state=SEED,max_iter=1000)
        elif name=='RandomForest':est=RandomForestClassifier(random_state=SEED,n_estimators=100,n_jobs=4)
        else:est=HistGradientBoostingClassifier(random_state=SEED,early_stopping=False,max_iter=100)
        return Pipeline([('preprocess',pre),('classifier',est)])
    def event_eval(frame,p,th,scenario,model,kind):
        f=frame.copy();f['alarm']=p>=th;positive=f[f.y.eq(1)];eligible=set(positive.episode_id.dropna());detected=set(f.loc[f.alarm&f.y.eq(1),'episode_id']);all_events=[]
        for tool in f.Tool.unique():
            b=bounds.loc[tool];start=b.first_time+.8*(b.last_time-b.first_time);all_events.extend(ep.loc[ep.Tool.eq(tool)&ep.event_start.ge(start)&ep.event_start.le(b.last_time),'episode_id'].tolist())
        for event,g in f[f.alarm&f.y.eq(1)].groupby('episode_id'):
            lead=(g.target_time.iloc[0]-g.anchor.min())/3600;assert lead>1 and lead<=24;leads.append(dict(scenario=scenario,model=model,threshold_kind=kind,episode_id=event,lead_hours=lead,Tool=g.Tool.iloc[0]))
        false_episodes=0;alarm_episodes=0;mixed=0
        for tool,g in f[f.alarm].sort_values(['Tool','anchor']).groupby('Tool'):
            group=g.anchor.diff().fillna(np.inf).gt(3600).cumsum()
            for _,a in g.groupby(group):
                alarm_episodes+=1;false_episodes+=int(a.y.eq(0).all());mixed+=int(a.y.nunique()>1)
        ll=[r['lead_hours'] for r in leads if r['scenario']==scenario and r['model']==model and r['threshold_kind']==kind]
        event_metrics.append(dict(scenario=scenario,model=model,threshold_kind=kind,eligible_events=len(eligible),all_test_events=len(set(all_events)),detected_events=len(detected),event_recall=len(detected)/len(eligible) if eligible else np.nan,all_event_detection_lower_bound=len(detected)/len(set(all_events)) if all_events else np.nan,event_coverage=len(eligible)/len(set(all_events)) if all_events else np.nan,alarm_episodes=alarm_episodes,false_alarm_episodes=false_episodes,mixed_alarm_episodes=mixed,false_alarm_rows=int((f.alarm&f.y.eq(0)).sum()),false_alarms_per_eligible_event=false_episodes/len(eligible) if eligible else np.nan,false_alarm_episodes_per_1000_evaluated_hours=1000*false_episodes/len(f),lead_min=np.min(ll) if ll else np.nan,lead_median=np.median(ll) if ll else np.nan,lead_p10=np.quantile(ll,.1) if ll else np.nan,lead_p90=np.quantile(ll,.9) if ll else np.nan,lead_max=np.max(ll) if ll else np.nan))
    def evaluate(frames,probs,scenario,name,th):
        for fold,f in frames.items():
            p=probs[fold];assert len(p)==len(f) and np.isfinite(p).all()
            for kind,t in [('fixed_0.5',.5),('validation_f1',th)]:
                metrics.append(dict(scenario=scenario,model=name,split=fold,threshold_kind=kind,threshold=t,**metric(f.y,p,t)))
            pred=f[['Tool','anchor','y','episode_id','target_time']].copy();pred['scenario']=scenario;pred['model']=name;pred['split']=fold;pred['probability']=p;pred['validation_threshold']=th;predictions.append(pred)
        test=frames['test'];p=probs['test'];bins=np.minimum((p*10).astype(int),9)
        for b in range(10):
            mask=bins==b
            if mask.any():calibration.append(dict(scenario=scenario,model=name,bin=b,n=int(mask.sum()),predicted=float(p[mask].mean()),observed=float(test.y.to_numpy()[mask].mean())))
        for kind,t in [('fixed_0.5',.5),('validation_f1',th)]:
            event_eval(test,p,t,scenario,name,kind)
            for tool,ii in test.groupby('Tool',sort=False).indices.items():
                sub=test.iloc[ii];device_metrics.append(dict(scenario=scenario,model=name,threshold_kind=kind,Tool=tool,unique_events=sub.loc[sub.y.eq(1),'episode_id'].nunique(),no_recorded_low_fault_ever=int(not ep.Tool.eq(tool).any()),**metric(sub.y,p[ii],t)))
            if scenario=='endpoint_only' and name in ['LogisticRegression','RandomForest','HistGradientBoosting'] and kind=='validation_f1':
                for process,ii in test.groupby(['recipe_mode','recipe_step_mode'],dropna=False,sort=False).indices.items():
                    sub=test.iloc[ii]
                    if len(sub)>=50:process_metrics.append(dict(model=name,recipe_mode=process[0],recipe_step_mode=process[1],tools=sub.Tool.nunique(),unique_events=sub.loc[sub.y.eq(1),'episode_id'].nunique(),**metric(sub.y,p[ii],t)))
    def run_scenario(frame,scenario,holdout=False,diagnostic=False):
        fs={fold:frame[frame.split.eq(fold)].copy().reset_index(drop=True) for fold in ['train','validation','test']}
        if holdout:
            fs['train']=fs['train'][~fs['train'].Tool.isin(held_tools)].reset_index(drop=True);fs['validation']=fs['validation'][~fs['validation'].Tool.isin(held_tools)].reset_index(drop=True);fs['test']=fs['test'][fs['test'].Tool.isin(held_tools)].reset_index(drop=True)
            assert not set(fs['train'].Tool)&set(fs['test'].Tool)
        for fold,f in fs.items():
            assert len(f) and f.y.nunique()==2,(scenario,fold,'insufficient classes')
            splits.append(dict(scenario=scenario,split=fold,n=len(f),negative=int(f.y.eq(0).sum()),positive=int(f.y.eq(1).sum()),positive_events=f.loc[f.y.eq(1),'episode_id'].nunique(),tools=f.Tool.nunique()))
        scenario_frames[scenario]=fs;w=weights(fs['train']);weight_records.append(dict(scenario=scenario,train_rows=len(w),negative_total=float(w[fs['train'].y.eq(0)].sum()),positive_total=float(w[fs['train'].y.eq(1)].sum()),min_weight=float(w.min()),max_weight=float(w.max())))
        todo=[('LogisticRegression',columns),('RandomForest',columns),('HistGradientBoosting',columns)]
        if diagnostic:todo=[('LogisticRegression_process_only',process_num+cats),('LogisticRegression_sensor_only',sensors),('HistGradientBoosting_process_only',process_num+cats)]
        for name,cols in todo:
            print('FIT',scenario,name,len(fs['train']),len(cols),flush=True);start=time.time();pipe=pipeline(name,cols)
            with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=4):
                warnings.simplefilter('always');pipe.fit(fs['train'][cols],fs['train'].y,classifier__sample_weight=w)
                probs={fold:pipe.predict_proba(f[cols])[:,1] for fold,f in fs.items()}
            warnings_log.extend(dict(scenario=scenario,model=name,message=str(z.message)) for z in caught)
            th=threshold(fs['validation'].y,probs['validation']);thresholds.append(dict(scenario=scenario,model=name,threshold=th,selection_split='validation',criterion='max F1; ties choose largest threshold'))
            evaluate(fs,probs,scenario,name,th);parameters.append(dict(scenario=scenario,model=name,fit_seconds=time.time()-start,features=cols,params=pipe.named_steps['classifier'].get_params()))
            joblib.dump(dict(pipeline=pipe,feature_columns=cols,validation_threshold=th,time_unit_assumption='seconds, unconfirmed',random_state=SEED),M/(scenario+'_'+name+'.joblib'),compress=3)
            if scenario=='endpoint_only' and not diagnostic:
                model_cache[name]=(pipe,cols);prob_cache[name]=probs
                names=pipe.named_steps['preprocess'].get_feature_names_out()
                if name=='LogisticRegression':values=pipe.named_steps['classifier'].coef_[0];kind='scaled_coefficient'
                elif name=='RandomForest':values=pipe.named_steps['classifier'].feature_importances_;kind='impurity_importance'
                else:values=None
                if values is not None:
                    for feature,value in zip(names,values):importance.append(dict(model=name,feature=feature,importance=float(value),method=kind))
            print('DONE',scenario,name,'threshold',th,'seconds',round(time.time()-start,1),flush=True)
        if diagnostic:return
        train=fs['train'];prior=float((train.y.sum()+1)/(len(train)+2))
        for name,keycols in [('AlwaysNormal',None),('GlobalPrior',[]),('ToolFrequency',['Tool']),('ProcessFrequency',cats)]:
            probs={}
            if keycols:
                sums=train.groupby(keycols,dropna=False).y.agg(['sum','size']);table_map={key if isinstance(key,tuple) else (key,):(r['sum']+20*prior)/(r['size']+20) for key,r in sums.iterrows()}
            for fold,f in fs.items():
                if name=='AlwaysNormal':probs[fold]=np.zeros(len(f))
                elif not keycols:probs[fold]=np.full(len(f),prior)
                else:probs[fold]=np.array([table_map.get(tuple(row),prior) for row in f[keycols].itertuples(index=False,name=None)])
            th=threshold(fs['validation'].y,probs['validation'],name=='AlwaysNormal');thresholds.append(dict(scenario=scenario,model=name,threshold=th,selection_split='validation' if name!='AlwaysNormal' else 'fixed',criterion='max F1' if name!='AlwaysNormal' else 'always label 0'))
            evaluate(fs,probs,scenario,name,th)
    run_scenario(endpoint,'endpoint_only')
    run_scenario(sensitivity,'observed50_no_gap6h')
    run_scenario(endpoint,'heldout_devices',holdout=True)
    run_scenario(endpoint,'feature_diagnostics',diagnostic=True)
    # Restrict the SAME endpoint models to the sensitivity test cohort, isolating evaluation selection.
    sf=scenario_frames['observed50_no_gap6h'];ef=scenario_frames['endpoint_only']
    selection=[]
    for name,(pipe,cols) in model_cache.items():
        pred=pipe.predict_proba(sf['test'][cols])[:,1];t=next(r['threshold'] for r in thresholds if r['scenario']=='endpoint_only' and r['model']==name)
        selection.append(dict(model=name,**metric(sf['test'].y,pred,t)))
    # HGB interpretation uses validation only, never test-based feature selection.
    pipe,cols=model_cache['HistGradientBoosting'];val=ef['validation'];sample=val.sample(n=min(2000,len(val)),random_state=SEED)
    print('VALIDATION PERMUTATION IMPORTANCE',len(sample),flush=True)
    with threadpool_limits(limits=4):perm=permutation_importance(pipe,sample[cols],sample.y,scoring='average_precision',n_repeats=2,random_state=SEED,n_jobs=1)
    for name,mean,std in zip(cols,perm.importances_mean,perm.importances_std):importance.append(dict(model='HistGradientBoosting',feature=name,importance=float(mean),std=float(std),method='validation_AP_permutation_drop'))
    tables={'metrics':metrics,'event_metrics':event_metrics,'early_warning_leads':leads,'device_metrics':device_metrics,'process_metrics':process_metrics,'calibration_bins':calibration,'split_counts':splits,'thresholds':thresholds,'feature_importance':importance,'training_weights':weight_records,'guard_exclusions':removed,'same_model_restricted_test':selection}
    for name,rows in tables.items():pd.DataFrame(rows).to_csv(L/(name+'.csv'),index=False)
    pd.concat(predictions,ignore_index=True).to_csv(L/'predictions.csv',index=False)
    (L/'model_parameters.json').write_text(json.dumps(parameters,default=str,indent=2),encoding='utf-8');(L/'warnings.json').write_text(json.dumps(warnings_log,indent=2),encoding='utf-8')
    (L/'input_features.json').write_text(json.dumps(dict(sensor_statistics=sensors,process_numeric=process_num,process_categorical=cats,excluded=[c for c in schema.column if c not in columns]),indent=2),encoding='utf-8')
    df[['Tool','anchor','followup','split','y','episode_id','target_time','window_start','label_end']].to_csv(L/'prepared_index.csv',index=False)
    summary=dict(random_state=SEED,time_unit='seconds assumption, not confirmed',heldout_tools=held_tools,feature_count=94,raw_reference_rows=len(idx),guarded_rows=len(df),old_files=len(json.loads((L/'preservation_hashes.json').read_text())),sklearn=sklearn.__version__,python=sys.version,final_model_selected=False)
    (L/'experiment_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    report(tables,summary,parameters,warnings_log)

def report(tables,summary,parameters,warnings_log):
    import numpy as np,pandas as pd,matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']=['Malgun Gothic','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    from sklearn.metrics import precision_recall_curve
    from html import escape
    def table(d):return d.to_html(index=False,border=0,float_format=lambda x:f'{x:.4g}',na_rep='해당 없음 / 평가 불가')
    def image(fig,name):
        fig.tight_layout();fig.savefig(F/(name+'.svg'));buf=io.BytesIO();fig.savefig(buf,format='png',dpi=130);plt.close(fig);return '<img alt="'+name+'" src="data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()+'">'
    d={key:pd.DataFrame(value) for key,value in tables.items()};metrics=d['metrics'];primary=['LogisticRegression','RandomForest','HistGradientBoosting'];pred=pd.read_csv(L/'predictions.csv');test=metrics[(metrics.scenario=='endpoint_only')&metrics.split.eq('test')]
    parts=['<h1>PHM2018 첫 머신러닝 모델 비교</h1><p>과거 센서·공정 정보로 미래 저압 고장 기록을 예측하는 첫 고정 설정 실험이다. 최종 모델 선정, 하이퍼파라미터 탐색, 서비스 구현은 수행하지 않았다. <b>time=초는 확정되지 않은 작업 가정</b>이다. 따라서 1시간·24시간 및 조기경보 선행시간의 물리적 의미는 이 가정에 종속된다.</p>']
    parts+=['<h2>학습 데이터와 표본 보존</h2><p>이전 설계 B_gap300(관측률 ≥80%, 최대·양끝 공백 ≤300단위, 센서 완전성 조건), 과거 [t−1h,t), 미래 (t+1h,t+24h], 고장 후 제외 6h, endpoint_only를 기준으로 사용했다. observed50_no_gap6h는 음성 라벨의 미래 관측 신뢰도 민감도다. 원본을 재표본화하거나 평가 결과에 따라 기준을 완화하지 않았다.</p><p>고장 후 제외가 입력 창 안에서도 지켜지도록 창 시작부터 직전 모든 유형 고장까지 >6h를 요구했다. 군집의 최초~마지막 시각이 60%/80% 경계를 가로지르면 그 사건 양성 창 전체를 제외했다. 이는 기존 설계를 느슨하게 한 것이 아니라 입력·군집 경계 보호를 강화한 것이다. 아래에 원래 유지 표본과 추가 제외 수를 공개한다.</p>',table(d['guard_exclusions']),'<h3>분할별 클래스·사건 수</h3>',table(d['split_counts'])]
    parts+=['<h2>분할·누수 검사</h2><p>장비별 첫~마지막 관측 기간 60/20/20 시간순 분할을 유지했다. [t−1h,t+24h]가 한 분할 안에 완전히 들어가는 창만 사용하므로 경계 앞 24h와 뒤 1h의 정보 중첩이 제거된다. 같은 목표 사건 ID가 서로 다른 유지 분할에 걸치지 않음을 확인했다. 주어진 학습/평가 데이터에서 임의 Random Split은 쓰지 않았다. 장비의 익명화 시계가 같은 달력 기준인지 미검증이므로 장비별 시간순 평가는 실제 동시점 배포와 같다고 주장하지 않는다.</p><p>모델 입력은 센서 17개×5통계(평균·표준편차·최솟값·최댓값·변화량) 85개와 과거 공정 구성 9개, 총 94개다. Tool, Lot, runnum, 절대 time, TTF, 미래 고장·센서·라벨 관련 감사 열, 관측 품질 감사값은 주 모델 입력에서 제외했다. 데이터 구성·라벨 감사에 쓰인 미래 정보와 모델 입력을 분리했다.</p>']
    inputs=json.loads((L/'input_features.json').read_text());parts+=['<details><summary>사용 특징 94개 전체 목록과 제외 열</summary><pre>'+escape(json.dumps(inputs,ensure_ascii=False,indent=2))+'</pre></details>']
    parts+=['<h2>전처리·모델·불균형 처리</h2><p>숫자 특징은 Train 중앙값으로 대체하고, Logistic Regression에만 Train 평균·표준편차 스케일링을 적용했다. 공정 최빈값 3개는 범주형 원핫 인코딩, 미관측 범주는 ignore 처리한다. 모든 전처리는 Pipeline에서 Train에만 fit했다. 상수 제거·특징 선택·확률 재보정은 하지 않았다.</p><p>학습 양성 창은 같은 사건 창 수의 역수로 가중해 사건별 총 영향이 같게 했다. 이후 두 클래스의 총 가중치가 각각 Train 행 수의 절반이 되도록 정규화했다. 음성은 클래스 안에서 동일 가중치다. 세 모델 모두 같은 sample_weight를 사용했다. 오버샘플링은 하지 않았고 Validation/Test에는 재가중하지 않았다. 가중 학습 확률은 실제 발생률로 바로 해석할 수 없어 아래 Calibration을 따로 확인한다.</p>',table(d['training_weights']),'<p>random_state=2018. LR은 기본 L2/C=1, max_iter=1000(수렴 여유), RF는 기본 100트리·깊이 제한 없음, HGB는 기본 100회이며 무작위 내부 검증 분할을 막기 위해 early_stopping=False로 고정했다. 서로 다른 하이퍼파라미터를 검색하지 않았다. 버전과 전체 파라미터는 requirements-model-lock.txt와 .local/model_comparison/model_parameters.json에 기록했다.</p>']
    compact=['model','threshold_kind','threshold','precision','recall','f1','average_precision','roc_auc','false_positive_rate','brier']
    parts+=['<h2>Test 성능: 0.5와 Validation 임계값</h2><p>Validation에서 F1이 최대인 임계값을 선택하고 동률이면 가장 큰 값을 택했다. Test의 확률·정답으로 임계값을 선택하지 않았다. F1 기준은 운용 경보 비용을 반영한 최종 임계값이 아니다. Average Precision을 PR-AUC로 보고했으며, 단순 사다리꼴 PR 면적과 구분한다. 단일 클래스 장비/구간의 ROC-AUC와 양성 0인 AP는 평가 불가로 표시했다. Accuracy는 참고 값이다.</p>',table(test[test.model.isin(primary)][compact]),'<details><summary>기준 모델 및 Confusion Matrix 수치 포함 전체 지표</summary>',table(test),'</details>']
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,fold in zip(axes,['validation','test']):
        for name in primary+['ToolFrequency','ProcessFrequency']:
            g=pred[(pred.scenario=='endpoint_only')&(pred.model==name)&(pred.split==fold)];pr,re,_=precision_recall_curve(g.y,g.probability);ax.plot(re,pr,label=name)
        g=pred[(pred.scenario=='endpoint_only')&(pred.model==primary[0])&(pred.split==fold)];ax.axhline(g.y.mean(),ls='--',color='gray',label='양성 비율');ax.set(title=fold+' PR Curve',xlabel='Recall',ylabel='Precision');ax.legend(fontsize=8)
    parts.append(image(fig,'pr_curves'))
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for row,kind in enumerate(['fixed_0.5','validation_f1']):
        for col,name in enumerate(primary):
            g=test[(test.model==name)&test.threshold_kind.eq(kind)].iloc[0];cm=np.array([[g.tn,g.fp],[g.fn,g.tp]],dtype=int);ax=axes[row,col];ax.imshow(cm,cmap='Blues');ax.set(title=name+' / '+kind,xlabel='예측',ylabel='실제',xticks=[0,1],yticks=[0,1])
            for i in range(2):
                for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    parts.append(image(fig,'confusion_matrices'))
    parts+=['<h2>기준 모델</h2><p>AlwaysNormal은 항상 0(확률 0)이다. GlobalPrior는 Train 발생 비율, ToolFrequency와 ProcessFrequency는 각각 Train 장비·공정 조합의 발생 빈도를 전역 비율로 20개 가상 관측만큼 수축했다. 미관측 그룹은 Train 전역 비율을 쓴다. 이 기준 모델의 Tool은 비교 목적일 뿐 주 모델 입력에는 없다. 빈도 기준도 Validation에서만 임계값을 택한다. 빈도 계산은 Train 정답만 사용하며 Train 자체 지표는 인표본 낙관성이 있을 수 있다.</p>']
    parts+=['<h2>사건 단위 조기경보·오경보</h2><p>eligible_events는 Test에 최소 한 개의 적격 양성 창이 있는 고유 운영상 저압 사건 군집이다. ≥1h·≤24h 이전 경보가 하나라도 있으면 탐지로 계산했다. 최초 경보부터 사건까지의 시간을 사건별 선행시간으로 보고한다. 최소 선행 1h 경계는 원래 라벨 정의상 엄격하게 초과한다. all_test_events는 각 Test 시간 구간의 모든 원래 사건 군집이며, 표본이 없는 사건을 포함한다. 두 탐지율을 구분하여 선택된 사건에만 좋은 결과를 전체 사건 성능으로 확대하지 않았다.</p><p>같은 장비에서 연속 시간격자 경보를 하나의 경보 episode로 합쳤다. 중간에 한 시간 이상 빈 격자가 있으면 끊는다. 전부 y=0인 경보 episode를 오경보로 정의하고, 혼합 episode 수와 오경보 행 수도 함께 보고한다. 한 episode가 일부 진양성을 포함하면 혼합으로 별도 표시되므로 행 단위 FP도 같이 봐야 한다. 오경보/사건은 적격 고유 사건 수가 분모다. 1,000시간당 오경보의 시간은 실제 연속 가동시간이 아니라 평가된 1시간 창 개수다. 사건 군집의 기계적 독립성은 미확인이다.</p>',table(d['event_metrics'][d['event_metrics'].scenario.eq('endpoint_only')])]
    fig,ax=plt.subplots(figsize=(10,5))
    for name in primary:
        g=d['early_warning_leads'][(d['early_warning_leads'].scenario=='endpoint_only')&(d['early_warning_leads'].model==name)&(d['early_warning_leads'].threshold_kind=='validation_f1')]
        if len(g):x=np.sort(g.lead_hours);ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',label=name+' n='+str(len(x)))
    ax.set(xlabel='최초 경보 선행시간 (초 단위 가정의 시간)',ylabel='누적 사건 비율',title='탐지된 사건의 선행시간 분포');ax.legend();parts.append(image(fig,'event_lead_ecdf'))
    parts+=['<h2>장비·공정 편중과 정상 장비 오경보</h2><p>no_recorded_low_fault_ever=1은 파일 전체에 저압 고장 기록이 없는 장비다. 다른 유형 고장이나 미기록 열화가 없다는 의미가 아니다. 해당 장비의 FPR=FP/(FP+TN)와 경보 수를 확인한다. Test 양성이 없는 장비의 AP/ROC-AUC는 성능이 좋다는 뜻으로 해석하지 않는다.</p>',table(d['device_metrics'][(d['device_metrics'].scenario=='endpoint_only')&d['device_metrics'].no_recorded_low_fault_ever.eq(1)]),'<details><summary>모든 장비별 지표</summary>',table(d['device_metrics'][d['device_metrics'].scenario.eq('endpoint_only')]),'</details>','<details><summary>공정별 지표: Test 50행 이상인 recipe/step 조합</summary><p>탐색 표이며 소수 양성·소수 사건의 높은 점수로 일반화하지 않는다. 집단 선택은 모델 조정에 사용하지 않았다.</p>',table(d['process_metrics']),'</details>']
    parts+=['<h2>과적합·공정 정보 진단</h2><p>Train/Test 차이를 그대로 공개한다. 센서만·공정만 사용하는 LR과 공정만 사용하는 HGB는 사전에 정한 진단 실험이며, Test 점수를 보고 주 모델의 특징을 다시 고르지 않았다. 공정만으로 점수가 높다면 물리적 고장 전조가 아니라 공정·장비 대리변수의 영향을 의심해야 한다.</p>',table(metrics[(metrics.scenario=='endpoint_only')&metrics.model.isin(primary)&metrics.threshold_kind.eq('validation_f1')]),table(metrics[(metrics.scenario=='feature_diagnostics')&metrics.threshold_kind.eq('validation_f1')])]
    parts+=['<h2>Calibration</h2><p>재보정 없이 원래 확률의 신뢰도를 평가했다. 동일한 10개 확률 구간의 평균 예측확률·관측 발생률·행 수를 보고하고 Brier와 Log loss도 비교한다. 클래스·사건 가중 학습은 확률을 실제 발생률과 다르게 만들 수 있다. 시간 및 사건 반복으로 각 bin의 행은 독립 표본이 아니므로 행 기반 신뢰구간을 제시하지 않았다.</p>']
    fig,axes=plt.subplots(1,2,figsize=(13,5));cal=d['calibration_bins'];cal=cal[cal.scenario.eq('endpoint_only')]
    for name in primary+['GlobalPrior','ToolFrequency','ProcessFrequency']:
        g=cal[cal.model.eq(name)];axes[0].plot(g.predicted,g.observed,marker='o',label=name)
    axes[0].plot([0,1],[0,1],ls='--',color='gray');axes[0].set(xlabel='평균 예측확률',ylabel='실제 관측 비율',title='Test reliability diagram');axes[0].legend(fontsize=8)
    for name in primary:
        g=pred[(pred.scenario=='endpoint_only')&(pred.model==name)&(pred.split=='test')];axes[1].hist(g.probability,bins=np.linspace(0,1,21),histtype='step',label=name)
    axes[1].set(xlabel='예측확률',ylabel='표본 수',title='Test 점수 분포');axes[1].legend();parts.append(image(fig,'calibration'))
    parts+=['<details><summary>Calibration bin 수치</summary>',table(cal),'</details>']
    parts+=['<h2>미관측 장비 Holdout</h2><p>장비명 정렬의 0·5·10·15번째 4대 '+', '.join(summary['heldout_tools'])+'를 사전에 기계적으로 지정했다. 이 장비의 모든 Train/Validation 행을 학습·전처리·임계값 선택에서 제외하고 후기 Test만 평가했다. 나머지 16대에서도 Train/Validation 시간순 분할을 유지했다. 각 장비의 상대 시간 평가이며 동기화된 달력 전향 평가와는 구분된다. 주 평가와 Test 장비 구성이 달라 숫자 차이를 순수한 일반화 손실로 단정하지 않는다. 4대 Holdout 한 번은 20회 leave-one-tool-out 검증의 대체가 아니다.</p>',table(metrics[(metrics.scenario=='heldout_devices')&metrics.split.eq('test')]),table(d['event_metrics'][d['event_metrics'].scenario.eq('heldout_devices')])]
    parts+=['<h2>라벨 관측 기준 민감도</h2><p>동일 파라미터로 observed50_no_gap6h의 Train/Validation/Test에서 다시 학습했다. 양성 정의는 같고 관측 부족 음성이 줄어 분포가 바뀐다. 별도 재학습 결과와, endpoint_only 모델을 그대로 둔 채 민감도 Test 집단에만 제한한 결과를 함께 제시한다. 표본 선택으로 AP가 높아진 것을 모델 개선으로 해석하지 않는다.</p>',table(metrics[(metrics.scenario=='observed50_no_gap6h')&metrics.split.eq('test')]),'<h3>기본 모델을 바꾸지 않고 Test 집단만 제한</h3>',table(d['same_model_restricted_test'])]
    parts+=['<h2>특징 중요도와 해석</h2><p>LR은 학습 스케일 계수, RF는 impurity 중요도(고유값이 많은 특징 편향 가능), HGB는 Validation 고정 표본 최대 2,000행에서 AP 감소 permutation 2회다. 중요도는 인과 증거가 아니고 강하게 상관된 센서 통계는 중요도를 나눠 가진다. 변수 섞기가 실제 물리 상태를 재현하지 않으며, 이 해석으로 모델을 재학습하지 않았다.</p>']
    for name in primary:
        g=d['feature_importance'][d['feature_importance'].model.eq(name)].copy();g['rank_value']=g.importance.abs() if name=='LogisticRegression' else g.importance;g=g.sort_values('rank_value',ascending=False).head(20);parts+=['<h3>'+name+'</h3>',table(g.drop(columns='rank_value'))]
    parts+=['<h2>첫 실험의 근거 기반 평가</h2>']
    for name in primary:
        g=test[test.model.eq(name)&test.threshold_kind.eq('validation_f1')].iloc[0];e=d['event_metrics'][(d['event_metrics'].scenario=='endpoint_only')&(d['event_metrics'].model==name)&(d['event_metrics'].threshold_kind=='validation_f1')].iloc[0]
        parts.append(f'<p>{name}: Test AP {g.average_precision:.4f}(Test 양성 비율 {g.prevalence:.4f}), Precision {g.precision:.4f}, Recall {g.recall:.4f}, F1 {g.f1:.4f}. 적격 사건 {int(e.eligible_events)}개 중 {int(e.detected_events)}개 탐지, 평가 1,000시간 창당 순수 오경보 episode {e.false_alarm_episodes_per_1000_evaluated_hours:.2f}개. 이 숫자들은 탐지율과 경보 부담을 함께 평가해야 한다.</p>')
    base=test[test.model.isin(['ToolFrequency','ProcessFrequency'])].average_precision.max();best=test[test.model.isin(primary)].average_precision.max();prev=test.prevalence.iloc[0]
    if best>max(base,prev):parts.append('<p>최소 한 모델의 관측된 AP가 빈도 기준과 무정보 양성 비율을 웃돌아 추가 연구의 탐색 근거는 있다. 그러나 이 결과만으로 신뢰할 수 있는 조기경보 시스템이나 최종 모델을 선정할 근거는 부족하다. 장비 Holdout, 공정 전용 진단, Calibration, 경보 부담과 함께 해석해야 한다.</p>')
    else:parts.append('<p>주 모델이 빈도 기준을 안정적으로 넘어선다는 근거가 부족하다. 현재 정의에서 운용 가능한 사전 예측이 가능하다고 주장할 수 없다. 이 실험은 실패 원인을 확인하는 연구 근거이며, 모델 복잡도를 늘리기 전 라벨·관측·공정 교란의 검증이 중요하다.</p>')
    parts+=['<h2>한계·보존·재현</h2><p>고장 라벨은 운전자의 정비 정지 시각이며 열화 시작은 알 수 없다. time 단위, 고장 로그 완전성, 정비 완료, 군집의 물리적 독립성은 미검증이다. 평가 가능 사건만의 탐지율과 모든 Test 사건 탐지 하한을 구분했다. 1시간 격자와 관측 품질 필터는 일부 운용 상태를 선택적으로 제외한다. 반복 사건·경보·동일 공정 때문에 표본 행은 독립적이지 않으며, 단일 시간 분할과 4대 Holdout의 점수에는 불확실성이 있다. Test 결과를 본 뒤 재학습하거나 기준을 완화하지 않았다. 확률은 검증된 실제 발생 확률이 아니다.</p><p>기존 파일의 해시를 비교했고 원본 센서·TTF를 이번 학습 단계에서 새로 읽거나 수정하지 않았다. 기존의 과거 특징 파일과 감사 라벨·작은 고장 기록만 읽었다. 모델·예측·환경 잠금·파라미터를 새 파일로 저장했다. models/model_comparison의 모델은 첫 실험 산출물이며 최종 배포 모델이 아니다. 모델 파일과 개인 표본은 Git 제외다. 최종 모델 선정·최적화·웹 대시보드는 진행하지 않았다.</p>', '<details><summary>실행 경고</summary><pre>'+escape(json.dumps(warnings_log,ensure_ascii=False,indent=2))+'</pre></details>', '<p><a href="classification_dataset_design.html">이전 데이터 설계</a> · <a href="timeseries_eda_report.html">시계열 EDA</a></p>']
    old=json.loads((L/'preservation_hashes.json').read_text());assert all(sha(P/rel)==h for rel,h in old.items())
    style='body{font-family:Malgun Gothic,sans-serif;max-width:1500px;margin:32px auto;padding:0 22px;color:#172d42;line-height:1.7}table{display:block;overflow:auto;border-collapse:collapse;font-size:12px}th,td{padding:7px 10px;border-bottom:1px solid #ddd;white-space:nowrap;text-align:right}th{background:#eaf1f8}img{width:100%}h2{margin-top:40px}pre{white-space:pre-wrap}'
    (R/'model_comparison_report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>PHM2018 첫 모델 비교</title><style>'+style+'</style>'+''.join(parts)+'</html>',encoding='utf-8')
    print('PRIMARY TEST\n'+test.to_string(index=False));print('EVENTS\n'+d['event_metrics'][d['event_metrics'].scenario.eq('endpoint_only')&d['event_metrics'].model.isin(primary)].to_string(index=False));print('REPORT COMPLETE',flush=True)

if __name__=='__main__':
    if args.report_only:
        import pandas as pd
        names=['metrics','event_metrics','early_warning_leads','device_metrics','process_metrics','calibration_bins','split_counts','thresholds','feature_importance','training_weights','guard_exclusions','same_model_restricted_test']
        report({name:pd.read_csv(L/(name+'.csv')) for name in names},json.loads((L/'experiment_summary.json').read_text()),json.loads((L/'model_parameters.json').read_text()),json.loads((L/'warnings.json').read_text()))
    elif not args.resume:
        prepare();subprocess.run([str(P/'.venv-model/Scripts/python.exe'),str(P/'src/model_comparison.py'),'--project',str(P),'--resume'],cwd=P,check=True)
    else:experiment()
