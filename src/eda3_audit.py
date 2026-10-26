"""Finish and audit only files created by the current stage 3 task."""
from pathlib import Path
import json,re,hashlib,subprocess,shutil,os
P=Path(__file__).resolve().parents[1]
L=P/'.local/eda3'
def patch_new():
    for name in ['eda3_stream.py','eda3_reports.py']:
        p=P/'src'/name
        text=p.read_text(encoding='utf-8')
        text=text.replace("P=Path(r'D:\\PHM2018-Predictive-Maintenance')","P=Path(__file__).resolve().parents[1]")
        text=text.replace("os.environ['YDATA_PROFILING_NO_ANALYTICS']='True'","os.environ['YDATA_PROFILING_NO_ANALYTICS']='True'")
        p.write_text(text,encoding='utf-8')
    gitignore=P/'.gitignore';text=gitignore.read_text(encoding='utf-8')
    if '.venv-eda/' not in text:gitignore.write_text(text+'\n# Separate stage-3 profiling environment\n.venv-eda/\n',encoding='utf-8')
    print('New EDA code path configuration and environment exclusions updated')

def audit():
    from html.parser import HTMLParser
    import xml.etree.ElementTree as ET
    import pandas as pd
    class Parse(HTMLParser):
        def __init__(self):super().__init__();self.images=[];self.links=[]
        def handle_starttag(self,tag,attrs):
            a=dict(attrs)
            if tag=='img':self.images.append(a.get('src',''))
            if tag=='a':self.links.append(a.get('href',''))
    checks={}
    for name in ['sensor_profiling_report.html','timeseries_eda_report.html']:
        p=P/'reports'/name;parser=Parse();text=p.read_text(encoding='utf-8');parser.feed(text)
        assert '<html' in text.lower() and len(text)>10000
        assert all(not x.startswith(('file:','http:','https:')) for x in parser.images)
        for ref in parser.links:
            if ref and not ref.startswith(('http:','https:','#','javascript:','mailto:','data:')):assert (p.parent/ref).is_file(),ref
        checks[name]=dict(bytes=p.stat().st_size,embedded_images=len(parser.images))
        svg_blocks=re.findall(r'<svg\b.*?</svg>',text,re.S)
        for block in svg_blocks:ET.fromstring(block)
        checks[name]['inline_svgs']=len(svg_blocks)
    for p in (P/'figures/eda3').glob('*.svg'):ET.parse(p)
    stats=pd.read_csv(L/'window_sensor_statistics.csv');good=stats[stats.n>0]
    assert (good.last_time<good.end).all() and (good.first_time>=good.start).all()
    assert ((good.end==good.event-good.lead_seconds)&(good.end-good.start==3600)).all()
    results=pd.read_csv(L/'event_matched_results.csv');assert not results.loc[results.usable,'contaminated'].any()
    for name in ['sampling_time_process_strata','sampling_process_strata']:
        s=pd.read_csv(L/(name+'.csv'));assert (s.sampled_rows>=1).all();assert s.population_rows.sum()==82189440
    population=pd.read_csv(L/'equipment_progress.csv');assert population.rows.sum()==82189440 and population.temporal_duplicate_rows_dropped.sum()==117533
    sample=pd.read_csv(L/'profiling_sample.csv');assert sample.Tool.nunique()==20
    assert not any('TTF' in c or 'fault' in c.lower() for c in sample.columns)
    old=json.loads((L/'old_hashes.json').read_text())
    assert all(hashlib.sha256((P/rel).read_bytes()).hexdigest()==h for rel,h in old.items())
    for path in ['.venv-eda/pyvenv.cfg','.local/eda3/profiling_sample.csv']:
        assert subprocess.run(['git','check-ignore','--no-index','-q',path],cwd=P).returncode==0
    # The user may have committed the initial project since environment setup.
    # Preserve that state; only the new private samples/environment must be untracked.
    assert not subprocess.check_output(['git','ls-files','--','.local/eda3','.venv-eda'],cwd=P,text=True).strip()
    for p in (P/'src').glob('eda3_*.py'):compile(p.read_text(encoding='utf-8'),str(p),'exec')
    checks.update(old_files_unchanged=len(old),time_window_checks='PASS',strata_checks='PASS',ignore_checks='PASS',model_training=False)
    frame=sample.drop(columns=['_row','_quartile'])
    checks['sample_rows']=len(frame)
    checks['sample_missing_cells']=int(frame.isna().sum().sum())
    checks['sample_duplicate_excess']=int(frame.duplicated().sum())
    (L/'audit.json').write_text(json.dumps(checks,indent=2),encoding='utf-8');print(json.dumps(checks,indent=2))

def enrich():
    import pandas as pd
    data=pd.read_csv(L/'event_matched_results.csv');data=data[data.usable & data.sensor.isin(['FLOWCOOLPRESSURE','FLOWCOOLFLOWRATE'])]
    rows=[]
    for (sensor,horizon),g in data.groupby(['sensor','horizon']):
        bytool=g.groupby('Tool').matched_difference.median()
        loo=[g.loc[g.Tool.ne(tool),'matched_difference'].median() for tool in bytool.index]
        rows.append(dict(sensor=sensor,horizon=horizon,events=len(g),tools=g.Tool.nunique(),largest_tool_share=g.Tool.value_counts().max()/len(g),pooled_event_median=g.matched_difference.median(),equal_tool_median=bytool.median(),without_03M02=g.loc[g.Tool.ne('03M02'),'matched_difference'].median(),leave_one_tool_out_min=min(loo),leave_one_tool_out_max=max(loo)))
    d=pd.DataFrame(rows);d.to_csv(L/'tool_influence_sensitivity.csv',index=False)
    table=d.to_html(index=False,border=0,float_format=lambda x:f'{x:.4g}',na_rep='미검증 / 없음')
    section='<h2>장비 집중과 반복 사건의 민감도</h2><p>동일 사건 가중 중앙값, 장비별 중앙값에 동일 가중치를 준 중앙값, 03M02 제외 값, 장비 하나씩 제외한 중앙값의 범위를 비교했다. largest_tool_share는 해당 선행시간·센서에서 가장 많은 사례를 제공한 장비의 비중이다. 장비 하나를 제외해 방향이 바뀌면 장비 구성에 민감한 탐색 결과로 해석한다. 이 비교는 재표본화 성능 평가나 독립성 보증이 아니다.</p>'+table
    p=P/'reports/timeseries_eda_report.html';text=p.read_text(encoding='utf-8');marker='<h2>근거와 결론</h2>'
    if '<h2>장비 집중과 반복 사건의 민감도</h2>' not in text:
        assert marker in text;text=text.replace(marker,section+marker);p.write_text(text,encoding='utf-8')
    all_data=pd.read_csv(L/'event_matched_results.csv');pressure=all_data[all_data.sensor.eq('FLOWCOOLPRESSURE')]
    sensitivity=[]
    for gap,coverage in [(60,.8),(300,.8),(300,.6)]:
        eligible=(pressure.coverage>=coverage)&(pressure.max_gap<=gap)&(pressure.first_time-pressure.start<=gap)&(pressure.end-pressure.last_time<=gap)&(~pressure.contaminated)&(pressure.matched_fraction>=.8)&(pressure.n>=100)
        for horizon,g in pressure[eligible].groupby('horizon'):
            sensitivity.append(dict(max_gap_seconds_assumed=gap,coverage_threshold=coverage,horizon=horizon,events=len(g),tools=g.Tool.nunique(),median_difference=g.matched_difference.median()))
    sensitivity=pd.DataFrame(sensitivity);sensitivity.to_csv(L/'gap_threshold_sensitivity.csv',index=False)
    if '<h2>측정 공백 기준 민감도</h2>' not in text:
        addition='<h2>측정 공백 기준 민감도</h2><p>비교 운용 구간은 기존 60초·80% 품질 기준을 유지하고, 고장 전 창의 최대 공백·양 끝 미관측 허용치를 300초로 완화한 경우를 비교했다. 주 분석은 60초·80% 기준을 유지했다. 이 완화 설정에서 관측시간 비율 80%와 60%의 결과는 동일했다. 표본을 늘리기 위한 최종 기준 선정이 아니며 정지 구간의 실제 의미는 미검증이다. 특히 1시간 전 압력 평균 차이의 사건 중앙값은 엄격 기준 −1.188에서 완화 기준 −0.03299로 크게 달라져, 사례 선택과 관측 공백 조건에 민감했다.</p>'+sensitivity.to_html(index=False,border=0,float_format=lambda x:f'{x:.4g}')
        text=p.read_text(encoding='utf-8').replace(marker,addition+marker);p.write_text(text,encoding='utf-8')
    sample=pd.read_csv(L/'profiling_sample.csv')
    temporal=sample.groupby(['Tool','_quartile']).size().unstack(fill_value=0).reset_index()
    temporal.to_csv(L/'sample_time_quartile_counts.csv',index=False)
    profile=P/'reports/sensor_profiling_report.html';text=profile.read_text(encoding='utf-8').replace('"xmlns:', '" xmlns:')
    if '<h2>표본의 상대 시간대 분포</h2>' not in text:
        addition='<h2>표본의 상대 시간대 분포</h2><p>0~3은 각 장비의 관측기간 4등분이다. 달력 날짜·교대조를 뜻하지 않는다. 층마다 행을 추출했기 때문에 원본 행 수 비율과는 다를 수 있다.</p>'+temporal.to_html(index=False,border=0)
        text=text.replace('</section>',addition+'</section>',1);profile.write_text(text,encoding='utf-8')
    else:profile.write_text(text,encoding='utf-8')
    print('TOOL INFLUENCE\n'+d.to_string(index=False))
if __name__=='__main__':
    import sys
    if '--patch' in sys.argv:patch_new()
    if '--enrich' in sys.argv:enrich()
    if '--audit' in sys.argv:audit()
