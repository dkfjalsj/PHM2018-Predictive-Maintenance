from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='4'
import sys,json,hashlib,subprocess,io,base64,re
import numpy as np,pandas as pd,joblib
from sklearn.metrics import precision_recall_curve,average_precision_score,confusion_matrix,brier_score_loss
from html.parser import HTMLParser
import xml.etree.ElementTree as ET
P=Path(sys.argv[1]).resolve();L=P/'.local/model_comparison';D=P/'.local/classification_design';PRIMARY=['LogisticRegression','RandomForest','HistGradientBoosting']
def read(name):return pd.read_csv(L/(name+'.csv'),float_precision='round_trip')
def audit():
    m=read('metrics');pred=read('predictions');splits=read('split_counts');th=read('thresholds');index=read('prepared_index');summary=json.loads((L/'experiment_summary.json').read_text());inputs=json.loads((L/'input_features.json').read_text());cols=inputs['sensor_statistics']+inputs['process_numeric']+inputs['process_categorical'];cats=inputs['process_categorical'];nums=[c for c in cols if c not in cats]
    assert len(cols)==94 and not {'Tool','anchor','label','target_time','episode_id','last_time','label_end','followup','split'}&set(cols)
    keys=index[['Tool','anchor']].drop_duplicates();parts=[]
    for x in pd.read_csv(D/'hour_features.csv',usecols=['Tool','anchor']+cols,chunksize=50000):parts.append(x.merge(keys,on=['Tool','anchor'],how='inner'))
    feature=pd.concat(parts,ignore_index=True)
    for c in cats:feature[c]=feature[c].map(lambda x:'__MISSING__' if pd.isna(x) else str(x)).astype(object)
    stats_checks=0
    for scenario in ['endpoint_only','observed50_no_gap6h','heldout_devices']:
        tr=pred[(pred.scenario==scenario)&pred.model.eq('LogisticRegression')&pred.split.eq('train')].merge(feature,on=['Tool','anchor'],validate='one_to_one')
        te=pred[(pred.scenario==scenario)&pred.model.eq('LogisticRegression')&pred.split.eq('test')]
        assert not set(tr.loc[tr.y.eq(1),'episode_id'])&set(te.loc[te.y.eq(1),'episode_id'])
        if scenario=='heldout_devices':assert not set(tr.Tool)&set(te.Tool)
        expected=np.nanmedian(tr[nums].to_numpy(),axis=0);expected=np.nan_to_num(expected,nan=0)
        for name in PRIMARY:
            bundle=joblib.load(P/'models/model_comparison'/(scenario+'_'+name+'.joblib'));pipe=bundle['pipeline'];pre=pipe.named_steps['preprocess'];imp=pre.named_transformers_['num'].named_steps['impute']
            assert bundle['feature_columns']==cols;np.testing.assert_allclose(imp.statistics_,expected,atol=1e-12)
            if name=='LogisticRegression':np.testing.assert_allclose(pre.named_transformers_['num'].named_steps['scale'].mean_,imp.transform(tr[nums]).mean(0),atol=1e-10)
            for j,c in enumerate(cats):assert set(pre.named_transformers_['cat'].categories_[j])==set(tr[c])
            assert pipe.named_steps['classifier'].random_state==2018
            if name=='HistGradientBoosting':assert pipe.named_steps['classifier'].early_stopping is False
            assert bundle['validation_threshold']==th[(th.scenario==scenario)&th.model.eq(name)].threshold.iloc[0]
            # Serialized predictions reproduce on a fixed test slice without fitting anything.
            own_test=pred[(pred.scenario==scenario)&pred.model.eq(name)&pred.split.eq('test')]
            test=own_test.head(50).merge(feature,on=['Tool','anchor'],validate='one_to_one');np.testing.assert_allclose(pipe.predict_proba(test[cols])[:,1],test.probability,atol=1e-12)
            stats_checks+=1
    for row in th.itertuples(index=False):
        if row.model=='AlwaysNormal':assert row.threshold==.5;continue
        v=pred[(pred.scenario==row.scenario)&pred.model.eq(row.model)&pred.split.eq('validation')]
        p,r,t=precision_recall_curve(v.y,v.probability);f=2*p[:-1]*r[:-1]/np.maximum(p[:-1]+r[:-1],1e-15);best=np.flatnonzero(np.isclose(f,f.max(),atol=1e-12,rtol=0));expected=t[best[-1]]
        assert np.isclose(row.threshold,expected,atol=1e-12)
    for row in m.itertuples(index=False):
        g=pred[(pred.scenario==row.scenario)&pred.model.eq(row.model)&pred.split.eq(row.split)];tn,fp,fn,tp=confusion_matrix(g.y,g.probability>=row.threshold,labels=[0,1]).ravel()
        assert (tn,fp,fn,tp)==(row.tn,row.fp,row.fn,row.tp)
        assert np.isclose(average_precision_score(g.y,g.probability),row.average_precision)
        assert np.isclose(brier_score_loss(g.y,g.probability),row.brier)
    assert not index.duplicated(['Tool','anchor','followup']).any()
    sys.path.insert(0,str(P));from src.config import data_root
    faults=pd.concat([pd.read_csv(p) for p in (data_root()/'train/train_faults').glob('*_train_fault_data.csv')],ignore_index=True)
    bounds=pd.read_csv(P/'reports/sensor_validation/equipment_validation.csv').set_index('Tool')
    for tool,g in index.groupby('Tool'):
        b=bounds.loc[tool];cut=[b.first_time,b.first_time+.6*(b.last_time-b.first_time),b.first_time+.8*(b.last_time-b.first_time),b.last_time]
        for i,fold in enumerate(['train','validation','test']):
            a=g[g.split.eq(fold)];assert (a.window_start>=cut[i]).all() and (a.label_end<=cut[i+1]).all()
        t=np.sort(faults.loc[faults.Tool.eq(tool),'time'].unique());ix=np.searchsorted(t,g.anchor.to_numpy(),side='right')-1;prior=np.full(len(g),-np.inf);ok=ix>=0;prior[ok]=t[ix[ok]];assert (g.window_start.to_numpy()-prior>21600).all()
        p=g[g.y.eq(1)];assert ((p.target_time>p.anchor+3600)&(p.target_time<=p.anchor+86400)).all()
    c=read('calibration_bins')
    for (scenario,name),g in c.groupby(['scenario','model']):
        test=pred[(pred.scenario==scenario)&pred.model.eq(name)&pred.split.eq('test')];assert g.n.sum()==len(test);assert np.isclose((g.n*g.observed).sum()/g.n.sum(),test.y.mean())
    e=read('event_metrics');assert (e.detected_events<=e.eligible_events).all() and (e.eligible_events<=e.all_test_events).all()
    lead=read('early_warning_leads');assert ((lead.lead_hours>1)&(lead.lead_hours<=24)).all()
    assert (read('training_weights').negative_total-read('training_weights').positive_total).abs().max()<1e-8
    for folder in ['.venv-model/pyvenv.cfg','.local/model_comparison/predictions.csv','models/model_comparison/endpoint_only_RandomForest.joblib']:
        assert subprocess.run(['git','check-ignore','--no-index','-q',folder],cwd=P).returncode==0
    hashes=json.loads((L/'preservation_hashes.json').read_text());assert all(hashlib.sha256((P/rel).read_bytes()).hexdigest()==h for rel,h in hashes.items())
    class Parse(HTMLParser):
        def __init__(self):super().__init__();self.images=[];self.links=[]
        def handle_starttag(self,tag,attrs):
            a=dict(attrs)
            if tag=='img':self.images.append(a.get('src',''))
            if tag=='a':self.links.append(a.get('href',''))
    report=P/'reports/model_comparison_report.html';parser=Parse();parser.feed(report.read_text(encoding='utf-8'));assert len(parser.images)>=4 and all(x.startswith('data:image/png;base64,') for x in parser.images)
    for p in (P/'figures/model_comparison').glob('*.svg'):ET.parse(p)
    assert all((report.parent/ref).is_file() for ref in parser.links if not ref.startswith(('http:','https:','#','data:')))
    result=dict(saved_pipeline_checks=stats_checks,train_only_imputation_encoding_scaling='PASS',validation_only_thresholds='PASS',metrics_recomputed='PASS',post_fault_and_temporal_purge='PASS',unique_event_fold_separation='PASS',holdout_devices_absent_from_training='PASS',calibration_counts='PASS',event_lead_boundaries='PASS',old_files_unchanged=len(hashes),ignored_private_models_and_environment='PASS',embedded_graphs=len(parser.images),final_model_selected=False)
    (L/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

def enhance():
    m=read('metrics');pred=read('predictions');c=read('calibration_bins');tab=[]
    for name in PRIMARY+['GlobalPrior','ToolFrequency','ProcessFrequency']:
        g=pred[(pred.scenario=='endpoint_only')&pred.model.eq(name)&pred.split.eq('test')];bins=c[(c.scenario=='endpoint_only')&c.model.eq(name)]
        tab.append(dict(model=name,mean_probability=g.probability.mean(),observed_frequency=g.y.mean(),brier=m[(m.scenario=='endpoint_only')&m.model.eq(name)&m.split.eq('test')].brier.iloc[0],ece10=(bins.n*(bins.predicted-bins.observed).abs()).sum()/bins.n.sum()))
    table=pd.DataFrame(tab);table.to_csv(L/'calibration_summary.csv',index=False)
    section='<h2>핵심 진단 해석</h2><p>RF는 Train AP 1.000에서 Validation 0.289, Test 0.069로, HGB는 0.936에서 0.240, 0.083으로 하락했다. LR도 0.327, 0.209, 0.051 순으로 낮아졌다. 과적합 또는 시간에 따른 분포 차이에 민감하다는 강한 경고이며 원인을 하나로 확정할 수는 없다.</p><p>공정 전용 LR/HGB의 Test AP는 0.0375/0.0327, 센서 전용 LR은 0.0489로, 전체 LR 0.0506과 비슷했다. 공정 정보만으로 주 성능이 설명된다는 근거는 약하지만, 센서에도 장비 대리 정보가 있을 수 있어 장비 식별 교란을 완전히 배제하지 못한다. Holdout 4대의 양성 사건은 20개뿐이다. RF는 Validation 임계값에서 Holdout 양성을 한 건도 잡지 못했고, 세 모델의 Holdout AP는 약 0.056~0.067로 그 집단 양성 비율 0.0467에 비해 제한적인 향상이다.</p><p>주 Test의 원래 저압 사건은 93개이고 적격 입력 창이 있는 사건은 56개(60.2%)다. Validation 임계값으로 탐지한 15~17개는 전체 93개 대비 16.1~18.3%다. 평가 가능한 사건만의 탐지율을 전체 사건으로 확대해서는 안 된다. 같은 임계값에서 1,000개 평가 시간 창당 순수 오경보는 약 46~60 episode다.</p><p>아래 평균 예측확률과 관측 비율의 차이, Brier와 ECE는 원래 확률의 과대/과소 예측을 보여준다. 세 모델의 Brier는 전역 발생 빈도 기준보다 나빴다. 가중 학습 점수를 실제 고장 확률로 해석할 근거가 부족하다. ECE는 10개 고정 구간의 빈도 가중 절대 차이이며, 별도 확률 재보정을 학습하지 않았다.</p>'+table.to_html(index=False,border=0,float_format=lambda x:f'{x:.4g}')
    p=P/'reports/model_comparison_report.html';text=p.read_text(encoding='utf-8');marker='<h2>첫 실험의 근거 기반 평가</h2>'
    if '<h2>핵심 진단 해석</h2>' not in text:text=text.replace(marker,section+marker);p.write_text(text,encoding='utf-8')

if __name__=='__main__':enhance();audit()
