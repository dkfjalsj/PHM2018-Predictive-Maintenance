"""Generate the requested YData and time-series EDA reports from bounded scan outputs."""
from pathlib import Path
import os, sys, json, hashlib, html, base64, io
os.environ['MPLBACKEND']='Agg'
os.environ['YDATA_PROFILING_NO_ANALYTICS']='True'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport

P=Path(__file__).resolve().parents[1];L=P/'.local/eda3';R=P/'reports';F=P/'figures/eda3'
HORIZONS=['1h','6h','24h','3d','7d']; SENSORS=pd.read_csv(L/'window_sensor_statistics.csv',nrows=0)
SOURCE='https://phmsociety.org/conference/annual-conference-of-the-phm-society/annual-conference-of-the-prognostics-and-health-management-society-2018-b/phm-data-challenge-6/'
plt.rcParams['font.family']=['Malgun Gothic','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
def table(d):return d.to_html(index=False,border=0,float_format=lambda x:f'{x:.4g}',na_rep='미검증 / 없음')
def image(fig,name):
    fig.tight_layout();fig.savefig(F/(name+'.svg'));b=io.BytesIO();fig.savefig(b,format='png',dpi=130);plt.close(fig)
    return '<img alt="'+name+'" src="data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()+'">'
def freezecheck():
    old=json.loads((L/'old_hashes.json').read_text()); changed=[]
    for rel,h in old.items():
        p=P/rel
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=h:changed.append(rel)
    assert not changed,changed
    return len(old)

def main():
    sample=pd.read_csv(L/'profiling_sample.csv'); pop=pd.read_csv(L/'equipment_progress.csv');windows=pd.read_csv(L/'window_sensor_statistics.csv');process=pd.read_csv(L/'window_process_statistics.csv');refs=pd.read_csv(L/'reference_process_hours.csv')
    for c in ['Tool','stage','Lot','recipe','recipe_step']:sample[c]=sample[c].astype('string')
    sensorcols=[c for c in sample.columns if c not in ['time','Tool','stage','Lot','runnum','recipe','recipe_step','_row','_quartile']]
    frame=sample.drop(columns=['_row','_quartile'])
    # Do not enable exploratory telemetry; HTML resources are inline/offline.
    profile=ProfileReport(frame,title='PHM2018 — 층화 표본 센서 프로파일',minimal=True,pool_size=2,progress_bar=False,correlations={'pearson':{'calculate':True},'spearman':{'calculate':True}},missing_diagrams={'bar':True,'matrix':True,'heatmap':False},duplicates={'head':10},samples={'head':5,'tail':5},html={'inline':True,'minify_html':False},explorative=False)
    profile.to_file(R/'sensor_profiling_report.html')
    desc=profile.get_description()
    # Attach transparent sampling methods ahead of the genuine YData report content.
    a=pd.read_csv(L/'sampling_time_process_strata.csv');b=pd.read_csv(L/'sampling_process_strata.csv')
    sampling=('<section style="padding:24px;background:#fff4d8;font-family:sans-serif"><h1>표본·해석 범위</h1><p>원본 전체가 아닌 장비·공정·시간 층화 표본 '+f'{len(frame):,}'+'행의 YData Profiling 결과입니다. 원본 82,189,440행은 100,000행씩 읽었습니다. 각 장비에서 ① 관측기간 4등분×recipe×recipe_step, ② stage×recipe×recipe_step의 각 관측 층마다 독립 난수 우선순위 최솟값 행을 하나씩 보존하고 두 표본의 합집합을 사용했습니다. 동일 원본 행만 합쳤으며 원본 중복 값은 사전에 삭제하지 않았습니다. seed=2018+장비번호입니다. 드문 공정이 과대표집되므로 전수 결측률·중복률·상관계수로 일반화하지 마세요.</p><p>시간대는 달력/교대조가 아니라 장비별 관측기간의 상대 4분위입니다. 시간대·기준 날짜는 확인되지 않았습니다. 원본의 매우 희귀한 결측이나 중복이 표본에 없을 수 있습니다. 기존 전수 검사에서는 결측 1,166셀, 동일 시각 간격 117,533개가 확인됐습니다. YData의 중복 행과 동일 시각 중복은 다른 정의입니다. 센서·시간·장비·공정 메타데이터만 포함하고 TTF 및 미래 고장 라벨은 포함하지 않았습니다. 상관관계는 장비/공정 혼합과 표본 설계의 영향을 받으며 인과·예측 성능이 아닙니다.</p>'+table(pop[['Tool','rows','sample_rows','time_recipe_step_strata','process_strata']])+f'<p>관측된 시간·공정 층 {len(a):,}개와 공정 층 {len(b):,}개에서 모두 최소 1개 표본이 존재함을 검사했습니다.</p></section>')
    assert (a.sampled_rows>=1).all() and (b.sampled_rows>=1).all()
    target=R/'sensor_profiling_report.html';text=target.read_text(encoding='utf-8').replace('"xmlns:', '" xmlns:');idx=text.find('>',text.lower().find('<body'))+1;assert idx>0;target.write_text(text[:idx]+sampling+text[idx:],encoding='utf-8')
    # Match only earlier, same-tool, identical process reference hours. The 7-day exclusion
    # surrounding each reference hour must already be observable at the prediction cutoff.
    keys=['Tool','stage','recipe','recipe_step','sensor']
    for d in [process,refs]:
        for c in ['stage','recipe','recipe_step']:d[c]=d[c].fillna(-999999).astype(float)
    grouped={k:g.sort_values('hour') for k,g in refs.groupby(keys,dropna=False)}
    matched=[]
    for row in process.itertuples(index=False):
        key=(row.Tool,row.stage,row.recipe,row.recipe_step,row.sensor);g=grouped.get(key)
        if g is None:continue
        # latest eligible reference; its full future label-exclusion period precedes prediction time.
        limit=(row.prediction_time-604800)/3600-1;ix=np.searchsorted(g.hour.to_numpy(),limit,side='right')-1
        if ix<0:continue
        r=g.iloc[ix];matched.append(dict(Tool=row.Tool,event=row.event,horizon=row.horizon,sensor=row.sensor,n=row.n,pre_mean=row.mean,control_mean=r['mean'],control_std=r['std'],control_hour=r.hour))
    matches=pd.DataFrame(matched)
    if len(matches):
        matches['weighted_pre']=matches.n*matches.pre_mean;matches['weighted_control']=matches.n*matches.control_mean
        matched_event=matches.groupby(['Tool','event','horizon','sensor']).agg(matched_n=('n','sum'),pre=('weighted_pre','sum'),control=('weighted_control','sum')).reset_index();matched_event['pre']/=matched_event.matched_n;matched_event['control']/=matched_event.matched_n;matched_event['matched_difference']=matched_event.pre-matched_event.control
    else:matched_event=pd.DataFrame(columns=['Tool','event','horizon','sensor','matched_n','pre','control','matched_difference'])
    data=windows.merge(matched_event,on=['Tool','event','horizon','sensor'],how='left');data['matched_fraction']=data.matched_n/data.n
    data['usable']=data.quality & ~data.contaminated & data.matched_fraction.ge(.8) & data.n.ge(100)
    data.to_csv(L/'event_matched_results.csv',index=False)
    counts=data[data.sensor.eq('FLOWCOOLPRESSURE')].groupby('horizon').agg(cluster_anchors=('event','size'),coverage_valid=('quality','sum'),uncontaminated=('contaminated',lambda x:(~x).sum()),usable=('usable','sum')).reindex(HORIZONS).reset_index()
    useful=data[data.usable];summary=useful.groupby(['horizon','sensor']).agg(events=('event','size'),tools=('Tool','nunique'),median_mean=('mean','median'),median_std=('std','median'),median_change=('change','median'),median_ma15=('trailing_ma15','median'),median_ma15_change=('ma15_change','median'),median_control_difference=('matched_difference','median'),positive_fraction=('matched_difference',lambda x:(x>0).mean())).reset_index()
    summary['horizon']=pd.Categorical(summary.horizon,HORIZONS,ordered=True);summary=summary.sort_values(['sensor','horizon'])
    bytool=useful.groupby(['Tool','horizon','sensor']).agg(events=('event','size'),median_difference=('matched_difference','median')).reset_index()
    for name,d in [('event_window_counts',counts),('sensor_horizon_summary',summary),('tool_horizon_summary',bytool)]:d.to_csv(L/(name+'.csv'),index=False)
    # Paired multi-horizon comparisons use the same event, not different case mixtures.
    pairrows=[]
    for sensor,g in useful.groupby('sensor'):
        pivot=g.pivot(index=['Tool','event'],columns='horizon',values='matched_difference')
        if '1h' not in pivot:continue
        for h in HORIZONS[1:]:
            if h not in pivot:continue
            pair=pivot[['1h',h]].dropna();diff=pair['1h']-pair[h]
            pairrows.append(dict(sensor=sensor,comparison='1h vs '+h,paired_events=len(pair),tools=pair.index.get_level_values('Tool').nunique(),median_paired_difference=diff.median() if len(pair) else np.nan))
    pairs=pd.DataFrame(pairrows);pairs.to_csv(L/'paired_horizon_comparisons.csv',index=False)
    # Sensor-only Pearson/Spearman outputs preserve the actual automated analysis basis.
    frame[sensorcols].corr(method='pearson').to_csv(L/'sample_pearson.csv');frame[sensorcols].corr(method='spearman').to_csv(L/'sample_spearman.csv')
    parts=['<h1>PHM 2018 3차 시계열 EDA — 냉각 압력 저하</h1><p><b>시간 단위는 초라는 작업 가정이다. 공식 설명은 고장 time을 “예: 초”로 표기하지만 확정적인 단위/기준 시각을 제공하지 않는다. 따라서 아래 1시간·6시간·24시간·3일·7일 해석은 이 가정이 맞을 때만 유효하다.</b> 센서의 물리 단위도 확인되지 않았다. 모델 학습·성능 평가·최종 예측 대상 선정은 하지 않았다.</p>',f'<p><a href="{SOURCE}">PHM Society 공식 데이터 설명</a>에 따르면 라벨은 정비를 위해 장비를 정지한 시각이다. 실제 열화 시작 시각은 제공되지 않는다. 비교 운용 구간도 물리적으로 정상임을 보장하지 않는다.</p>']
    parts+=['<h2>분석 정의와 누수 방지</h2><p>저압 원본 756행에서 동일 장비·시각·유형 중복을 제거해 741개 조합을 얻었다. 같은 장비의 저압 고장 간격 ≤1시간(초 가정)이면 연결 군집으로 묶고 최초 시각을 대표로 사용했다. 다른 병합 기준의 민감도도 아래에 제시한다. 수리 이력이 없어 군집이 독립 고장 사건임을 보장하지 않는다.</p><p>고장시각 F, 선행시간 H에 대해 통계창은 [F−H−1시간,F−H)다. H=24시간 통계에는 F−24시간 이후 센서가 전혀 들어가지 않는다. 고장 시각·TTF·미래 고장 거리는 센서 입력에 추가하지 않았다. 미래 고장 정보는 오직 사후 사건 정의와 EDA 비교군 배제에 썼다. 시계열 통계는 같은 time의 첫 행을 남기고 계산했으며, 최초 기록 선택이 동시 측정의 완전한 해결책임을 뜻하지는 않는다.</p><p>평균=유효 관측값 합/n, 표준편차는 ddof=1, 변화량=창 마지막−첫 값, 이동평균=예측 기준시점 직전 15분 관측값 평균, 이동평균 변화=직전 15분 평균−그 앞 15분 평균이다. 이동평균은 현재·과거만 포함한다. 시간가중 평균은 아니며 공정 전환 시 첫/끝 차이에 교란이 있을 수 있다.</p><p>공백을 보간하지 않았다. 1시간 창은 관측 간격을 최대 4초로 제한해 합친 관측시간 비율 ≥80%, 창 내부 최대 공백 ≤60초, 양 끝 미관측 ≤60초, 센서 유효값 ≥100개일 때만 사용했다. nominal 4초는 기존 전수 측정간격 중앙값이다. 60초 등 기준도 시간 단위 가정에 종속된다. 창 시작부터 목표 고장 직전 사이에 다른 유형을 포함한 별도 고장시각이 있으면 오염으로 제외했다. 근접 사건이 많은 장비와 긴 선행시간에서 사례가 줄어드는 선택 효과가 생긴다.</p><p>비교군은 같은 장비·stage·recipe·recipe_step에서, 전후 7일 내 어느 고장 기록도 없는 품질 통과 1시간 구간이다. 그 구간의 후속 7일까지가 해당 예측 기준시점 이전에 끝나도록 제한한 뒤 가장 가까운 과거 구간을 골랐다. 즉 예측시점 이후의 센서·비교군을 쓰지 않는다. 다만 사후 고장 기록으로 정상 비교군을 선정했으므로 실시간 정상 판정 규칙은 아니다. 비교군은 sensor 유효값 ≥30개, 고장 전 공정별 유효값 ≥30개를 요구하고, 대응되는 공정이 창 센서 관측의 ≥80%를 덮을 때만 비교했다. 공정 비중으로 비교 평균을 가중했다. 비교군 재사용·더 오래된 시기와의 장기 드리프트는 남는 한계다.</p>']
    parts+=['<h2>표본과 사건 수</h2>',table(pop),table(counts),'<h3>저압 군집 민감도</h3>',table(pd.read_csv(L/'cluster_sensitivity.csv').groupby('threshold_seconds_assumed',as_index=False)[['unique_records','clusters']].sum())]
    fig,ax=plt.subplots(figsize=(10,4));ax.bar(counts.horizon,counts.coverage_valid,label='관측 품질 통과');ax.bar(counts.horizon,counts.usable,label='공정 대응·오염 제외 후');ax.set(ylabel='대표 사건 수',title='선행시간별 관측·비교 가능 사건');ax.legend();parts.append(image(fig,'event_counts'))
    parts+=['<h2>센서별 사건 단위 통계</h2><p>각 행은 사건 통계를 먼저 계산한 뒤 사건에 동일 가중치를 준 중앙값이다. 행 수가 많은 사건이 더 큰 가중치를 받지 않는다. 선행시간별 서로 다른 사건이 포함되므로, 전체 중앙값 변화만으로 열화 궤적을 단정하지 않는다.</p>',table(summary)]
    for sensor in ['FLOWCOOLPRESSURE','FLOWCOOLFLOWRATE','IONGAUGEPRESSURE']:
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        for ax,metric,title in zip(axes,['mean','std','trailing_ma15'],['평균','표준편차','직전 15분 이동평균']):
            arrays=[useful.loc[(useful.sensor==sensor)&(useful.horizon==h),metric].dropna().to_numpy() for h in HORIZONS]
            ax.boxplot([x if len(x) else [np.nan] for x in arrays],tick_labels=HORIZONS,showfliers=False);ax.set(title=title,ylabel='원본 센서 스케일')
        fig.suptitle(sensor+' — 각 선행시간의 직전 1시간 창');parts.append(image(fig,sensor+'_window_metrics'))
        fig,ax=plt.subplots(figsize=(11,5));g=bytool[bytool.sensor.eq(sensor)]
        for tool,gg in g.groupby('Tool'):
            vals=gg.set_index('horizon').median_difference.reindex(HORIZONS);ax.plot(HORIZONS,vals,marker='o',alpha=.65,label=tool)
        ax.axhline(0,color='black',lw=1);ax.set(title=sensor+' — 장비별 공정 대응 평균 차이',ylabel='고장 전 − 과거 비교 운용');ax.legend(ncol=4,fontsize=8,loc='upper left',bbox_to_anchor=(1,1));parts.append(image(fig,sensor+'_tool_differences'))
    parts+=['<h2>장비 차이·동일 사건 반복 비교</h2>',table(bytool),'<h3>같은 사건의 1시간 전과 더 긴 선행시간 차이</h3>',table(pairs),'<p>짝지은 사건 수와 장비 수를 함께 확인해야 한다. 사건 겹침과 비교군 재사용으로 독립 관측이 아니며, 다중 센서·선행시간 탐색에 대해 유의성 검정 또는 검증된 경보 성능을 주장하지 않았다. 병합 임계값 변경은 위 건수만 비교했고 모든 설정에서 센서 통계를 재추정한 민감도 검증은 아직 수행하지 않았다.</p>']
    pressure=summary[summary.sensor.eq('FLOWCOOLPRESSURE')]
    parts+=['<h2>근거와 결론</h2>',table(pressure),'<p><b>고장 전 예측 가능한 징후는 미검증이다.</b> 본 분석에서 관측된 공정 대응 평균 차이·변동성·이동평균은 탐색 근거다. 장비별 효과 방향, 동일 사건의 여러 시점 비교, 사용 가능한 사건 수를 함께 제시했으며, 공정 대응 실패·긴 공백·이전 고장 배제로 선택된 사례만 남는다. 공정 내 상태·Lot·장기 추세·정비 복구 상태를 완전히 통제하지 못했다. 반복 사건 및 창 겹침도 완전히 제거되지 않았다. 독립 장비·미래 기간에서의 재현성과 거짓 경보·선행시간별 성능을 검증하지 않았으므로 이 결과를 모델 성능이나 운용 가능한 고장 경보로 해석하지 않는다.</p>']
    nold=freezecheck();parts += [f'<h2>재현·보존</h2><p>기존 코드·보고서·그래프 {nold}개 파일의 SHA-256 일치를 확인했다. 센서 원본은 읽기 전후 크기·수정시각을 비교했다. 원본·기존 분석 결과를 수정하지 않았고 TTF 원본은 이번 센서 EDA 입력으로 읽지 않았다. 청크 100,000행, 난수 seed 2018+장비번호. 상세 표본·통계·환경 잠금 파일은 .local/eda3와 requirements-eda-lock.txt에 있다. 신규 센서 원본 표본·중간 CSV는 기본 Git 제외 영역에만 저장했다. 모델은 학습하지 않았다.</p><p><a href="sensor_profiling_report.html">YData Profiling 보고서</a> · <a href="sensor_validation/sensor_validation_report.html">이전 센서 검증</a> · <a href="fault_analysis/report.html">이전 고장 분석</a></p>']
    style='body{font-family:Malgun Gothic,sans-serif;max-width:1500px;margin:32px auto;padding:0 22px;color:#172d42;line-height:1.7}table{display:block;overflow:auto;border-collapse:collapse;font-size:12px}th,td{padding:7px 10px;border-bottom:1px solid #ddd;white-space:nowrap;text-align:right}th{background:#eaf1f8}img{width:100%;height:auto}h2{margin-top:40px}'
    (R/'timeseries_eda_report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>PHM2018 3차 시계열 EDA</title><style>'+style+'</style>'+''.join(parts)+'</html>',encoding='utf-8')
    pd.DataFrame([dict(sample_rows=len(frame),original_rows=int(pop.rows.sum()),preserved_old_files=nold,model_trained=False)]).to_json(L/'eda_summary.json',orient='records',indent=2)
    print('COUNTS\n',counts.to_string(index=False));print('PRESSURE\n',pressure.to_string(index=False));print('PAIRS\n',pairs[pairs.sensor.eq('FLOWCOOLPRESSURE')].to_string(index=False));print('REPORTS COMPLETE',flush=True)
if __name__=='__main__':main()
