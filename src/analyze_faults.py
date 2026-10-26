from pathlib import Path
from src.config import PROJECT_ROOT, data_root, output_dir, report_workdir
import pandas as pd
import numpy as np
import hashlib, json, html

ROOT=data_root()/'train'/'train_faults'
OUT=output_dir('fault_analysis')
files=sorted(ROOT.glob('*.csv'))
frames=[]; manifest=[]; checks=[]
for p in files:
    b=p.read_bytes(); d=pd.read_csv(p, dtype=str,keep_default_na=False)
    assert set(d.columns)=={'time','fault_name','Tool'}, (p,d.columns)
    manifest.append(dict(file=p.name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),rows=len(d)))
    d['source_file']=p.name; d['source_row']=np.arange(len(d))+2
    expected=''.join(p.stem.split('_')[:2])
    t=pd.to_numeric(d.time,errors='coerce')
    checks.append(dict(file=p.name,rows=len(d),missing_cells=int(d[['time','fault_name','Tool']].apply(lambda s:s.str.strip().eq('')).sum().sum()),non_numeric_or_nonfinite=int((~np.isfinite(t)).sum()),negative_time=int((t<0).sum()),fractional_time=int((t.notna() & np.isfinite(t) & t.mod(1).ne(0)).sum()),backward_steps=int((t.diff()<0).sum()),tool_filename_mismatch=int(d.Tool.ne(expected).sum()),whitespace_cells=int(d[['time','fault_name','Tool']].apply(lambda s:s.ne(s.str.strip())).sum().sum())))
    frames.append(d)
df=pd.concat(frames,ignore_index=True); df['t']=pd.to_numeric(df.time,errors='coerce')
valid=df[np.isfinite(df.t)].copy()
tools=df.groupby('Tool').size().rename('fault_count').to_frame()
types=df.groupby('fault_name').agg(fault_count=('Tool','size'),tools=('Tool','nunique')).sort_values('fault_count',ascending=False)
types['share_pct']=types.fault_count/len(df)*100
cross=pd.crosstab(df.Tool,df.fault_name)
intervals=[]; same=[]
for key,g in valid.groupby('Tool'):
    g=g.sort_values(['t','source_file','source_row'])
    for a,b in zip(g.to_dict('records'),g.to_dict('records')[1:]):
        intervals.append(dict(Tool=key,previous_time=a['t'],time=b['t'],gap=b['t']-a['t'],previous_fault=a['fault_name'],fault_name=b['fault_name'],source_file=b['source_file'],source_row=b['source_row']))
for (tool,fault),g in valid.groupby(['Tool','fault_name']):
    t=g.t.sort_values(); gaps=t.diff().dropna()
    same.append(dict(Tool=tool,fault_name=fault,n=len(t),interval_n=len(gaps),min_gap=gaps.min(),median_gap=gaps.median(),mean_gap=gaps.mean(),max_gap=gaps.max()))
gaps=pd.DataFrame(intervals)
time_summary=valid.groupby('Tool').agg(first_time=('t','min'),last_time=('t','max'))
time_summary['span']=time_summary.last_time-time_summary.first_time
stats=gaps.groupby('Tool').gap.agg(interval_n='size',min_gap='min',median_gap='median',mean_gap='mean',max_gap='max')
summary=tools.join(time_summary).join(stats)
dup=df.duplicated(['Tool','time','fault_name'],keep=False)
extra=df.duplicated(['Tool','time','fault_name']).sum()
numeric_dup=valid.duplicated(['Tool','t','fault_name'],keep=False)
ties=valid.groupby(['Tool','t']).agg(rows=('fault_name','size'),types=('fault_name','nunique')).query('rows>1')
audit=pd.DataFrame(checks)
quality={'files':len(files),'rows':len(df),'tools':df.Tool.nunique(),'fault_types':df.fault_name.nunique(),'duplicate_rows_involved':int(dup.sum()),'duplicate_excess':int(extra),'numeric_duplicate_rows_involved':int(numeric_dup.sum()),'same_tool_time_groups':len(ties),'same_tool_time_multiple_types':int((ties.types>1).sum()),'missing_cells':int(audit.missing_cells.sum()),'nonfinite_time':int(audit.non_numeric_or_nonfinite.sum()),'negative_time':int(audit.negative_time.sum()),'backward_steps':int(audit.backward_steps.sum()),'zero_gaps':int(gaps.gap.eq(0).sum()),'min_time':float(valid.t.min()),'max_time':float(valid.t.max())}
tables={'equipment_summary':summary,'fault_type_counts':types,'equipment_fault_crosstab':cross,'same_fault_intervals':pd.DataFrame(same),'file_quality_checks':audit,'source_manifest':pd.DataFrame(manifest),'consecutive_fault_intervals':gaps,'duplicate_records':df.loc[dup].drop(columns='t'),'same_time_groups':ties}
for name,tab in tables.items(): tab.to_csv(OUT/(name+'.csv'),encoding='utf-8-sig',index=name in ['equipment_summary','fault_type_counts','equipment_fault_crosstab','same_time_groups'])
(OUT/'quality.json').write_text(json.dumps(quality,ensure_ascii=False,indent=2),encoding='utf-8')
def svg(name,title,body,w=1400,h=780):
    (OUT/(name+'.svg')).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><rect width="100%" height="100%" fill="white"/><g font-family="Malgun Gothic,sans-serif" fill="#172d42"><text x="30" y="35" font-size="23">{html.escape(title)}</text>{body}</g></svg>',encoding='utf-8')
def bars(series,title,name):
    body=''
    for i,(label,value) in enumerate(series.items()):
        y=75+i*30; width=700*value/series.max()
        body+=f'<text x="10" y="{y+17}" font-size="13">{html.escape(str(label))}</text><rect x="470" y="{y}" width="{width}" height="22" fill="#2563a6"/><text x="{480+width}" y="{y+17}" font-size="13">{value}</text>'
    svg(name,title,body,h=max(230,110+len(series)*30))
bars(tools.fault_count,'장비별 고장 기록 수 (건)','equipment_counts')
bars(types.fault_count,'고장 유형별 기록 수 (건)','counts')
body=''
for j,label in enumerate(cross.columns): body+=f'<text x="{170+j*400}" y="75" font-size="12">{html.escape(label)}</text>'
for i,(tool,row) in enumerate(cross.iterrows()):
    y=100+i*29; body+=f'<text x="25" y="{y+20}">{tool}</text>'
    for j,value in enumerate(row):
        opacity=.08+.85*value/cross.values.max()
        body+=f'<rect x="{150+j*400}" y="{y}" width="390" height="27" fill="#2563a6" fill-opacity="{opacity}"/><text x="{330+j*400}" y="{y+20}" fill="black">{value}</text>'
svg('crosstab','장비 × 고장 유형 (건)',body)
body=''
lo,hi=valid.t.min(),valid.t.max()
for i,(tool,g) in enumerate(valid.groupby('Tool')):
    y=85+i*29; body+=f'<text x="10" y="{y+5}">{tool}</text><line x1="100" y1="{y}" x2="1300" y2="{y}" stroke="#ddd"/>'
    for t in g.t: body+=f'<circle cx="{100+1200*(t-lo)/(hi-lo)}" cy="{y}" r="3" fill="#2563a6" fill-opacity=".65"/>'
body+=f'<text x="100" y="700">{lo:g}</text><text x="1150" y="700">{hi:g}</text><text x="450" y="735">원본 time (단위 미확인)</text>'
svg('timing','장비별 고장 시점',body)
body=''
maxgap=gaps.gap.max()
for i,(tool,g) in enumerate(gaps.groupby('Tool')):
    y=85+i*29; body+=f'<text x="10" y="{y+5}">{tool}</text><line x1="100" y1="{y}" x2="1300" y2="{y}" stroke="#ddd"/>'
    for gap in g.gap: body+=f'<circle cx="{100+1200*gap/maxgap}" cy="{y}" r="3" fill="#368f82" fill-opacity=".5"/>'
    med=g.gap.median(); x=100+1200*med/maxgap
    body+=f'<line x1="{x}" x2="{x}" y1="{y-9}" y2="{y+9}" stroke="#c33" stroke-width="3"/>'
body+=f'<text x="100" y="700">0</text><text x="1150" y="700">{maxgap:g}</text><text x="400" y="735">원본 time 차이 (단위 미확인), 빨강: 중앙값</text>'
svg('gaps','장비별 연속 고장 간격 분포',body)
def tbl(x): return x.to_html(border=0,float_format=lambda v:f'{v:,.2f}',na_rep='해당 없음')
sections=[]
sections.append('<h1>PHM 2018 고장 기록 분석</h1><p>분석 범위: train_faults의 CSV 20개만. 모델 학습 및 예측 대상 최종 선정은 수행하지 않음.</p>')
sections.append(f'<p>원본 위치: {html.escape(str(ROOT))}</p><h2>계산 기준과 해석</h2><p>헤더를 제외한 CSV 한 행을 고장 기록 1건으로 계산했다. 중복도 임의 삭제하지 않은 원본 기준이다. 장비는 Tool 열로 구분하며 파일명과 대조했다. 고장 유형은 fault_name의 원문 문자열로 집계했다. 비율은 유형별 행 수 ÷ 전체 행 수 × 100이다. 교차표 각 셀은 해당 Tool과 fault_name을 동시에 만족하는 행 수이며, 행·열 합계를 개별 집계와 대조했다.</p><p>time에는 숫자만 있으며 이 CSV 자체에는 단위·기준 시각·시간대 설명이 없다. 따라서 달력 날짜나 초·일로 변환하지 않았다. 최초·마지막 시점은 장비별 min(time), max(time), span은 그 차이다. 장비 내부에서 time을 오름차순 정렬한 뒤 인접 시각 차이를 구했다. N건의 유효 시각에서 간격은 N−1개다. 평균은 간격 합÷간격 수, 중앙값은 정렬된 간격의 중앙값이다. 서로 다른 장비 사이의 시간 차이는 계산하지 않았다. 동일 유형 반복 간격은 장비와 fault_name으로 함께 묶어 동일 방식으로 계산했다.</p><p>이 간격은 기록 사이의 차이이며 실제 가동시간 기준 MTBF나 수리 후 재고장 시간으로 해석할 수 없다. 장비별 관측 시작·종료, 정상 구간, 수리 이력은 이번 범위에서 확인하지 않았다. 장비 간 기록 수 차이 역시 고장률 차이를 의미하지 않는다.</p>')
sections.append('<h2>전체 및 품질 검사</h2>'+tbl(pd.DataFrame(list(quality.items()),columns=['검사항목','결과'])))
sections.append('<p>중복은 (Tool, time 원문, fault_name)의 동일 조합으로 검사했으며, 숫자로 변환한 time 기준 중복도 별도 검사했다. 같은 장비·같은 시각에 여러 유형이 기록된 경우는 동시 고장 후보로 구분했다. 누락은 빈 문자열·공백 문자열, 시간 이상은 숫자 변환 불가/무한대·음수·소수·파일 원래 순서의 역행으로 검사했다. 매우 큰 시각이나 긴 간격은 단위와 관측 범위가 없어 오류로 단정하지 않았다. 실제 train 센서 시각 범위와의 정합성은 검사 범위 밖이다.</p>'+tbl(audit))
sections.append('<h2>장비별 집계 및 시간 통계</h2>'+tbl(summary)+'<h2>유형별 집계</h2>'+tbl(types)+'<img src="equipment_counts.svg"><img src="counts.svg"><h2>교차표</h2>'+tbl(cross)+'<img src="crosstab.svg"><h2>시점 및 반복 간격</h2><img src="timing.svg"><img src="gaps.svg">'+tbl(pd.DataFrame(same)))
dedup=df.drop_duplicates(['Tool','time','fault_name'])
sensitivity=types[['fault_count']].rename(columns={'fault_count':'original_rows'}).join(dedup.groupby('fault_name').size().rename('unique_records'))
sensitivity['duplicate_excess']=sensitivity.original_rows-sensitivity.unique_records
sensitivity.to_csv(OUT/'duplicate_sensitivity.csv',encoding='utf-8-sig')
sections.append('<h2>중복의 영향</h2>'+tbl(sensitivity)+'<p>동일 조합 중복 25그룹은 각 2행으로, 관련 행은 50행이고 첫 행만 남기는 가정에서 제외되는 행은 25행이다. 중복을 제외하면 총 1,211개 조합이며 장비 내부 전체 유형 간격 수는 1,191개가 된다. 주 집계는 원본 1,236행과 간격 1,216개를 유지했다. 0 간격 25개는 모두 동일 조합 중복에서 발생했다. 중복이 수집 중복인지 실제 복수 기록인지는 판정하지 않았다.</p>')
sections.append('<h2>예측 대상 후보 (최종 선정 아님)</h2><p>아래 순서는 사례 수 기준 검토 순서이며 통계적으로 충분한 표본 수를 보장하지 않는다.</p>'+tbl(types))
for fault,r in types.iterrows():
    sections.append(f'<p><b>{html.escape(fault)}</b>: {int(r.fault_count)}건, {int(r.tools)}개 장비, 전체의 {r.share_pct:.2f}%. 사례 수가 많을수록 우선 검토하기 쉽지만 장비별 편중과 독립 사건 여부를 확인해야 한다.</p>')
sections.append('<p>[NoWaferID]가 붙은 유형은 03M02에서 1건뿐이므로 독립 예측 유형으로는 사례 수가 매우 제한적이다. 일반 고압 유형 340건과의 의미상 관계를 확인하기 전에는 병합하지 않는다. 중복 제외 시의 후보별 건수는 위 민감도 표를 참조한다. 저압 유형 중 03M02에 301건이 몰려 있어 장비별 편중도 고려해야 한다.</p><p>추가 후보로 모든 유형을 합친 “임의 고장 발생” 이진 예측을 검토할 수 있다. 전체 기록 수는 양성 라벨 후보 수일 뿐 독립 학습 표본 수가 아니다. 다중 유형 분류 역시 후보이나 가장 적은 유형의 사례 수와 장비별 분포가 제약이다. 정상/음성 구간, 센서 연결, 예측 선행 시간, 사건 병합 기준 및 장비·시간 분할이 확인되지 않아 성능이나 최종 대상을 결정하지 않았다.</p><h2>Git 제외 설정</h2><p>프로젝트 루트는 현재 Git 저장소가 아니고 기존 .gitignore가 없어 새로 추가했다. 데이터 폴더(train, test, train_faults, phm_data_challenge_2018 접두 폴더), 원본 파일명 패턴과 압축파일을 제외한다. D드라이브 원본은 프로젝트 밖에 있고 이번 작업에서 복사하지 않았다. 향후 Git 저장소에 이미 추적 중인 파일이 있다면 .gitignore만으로 추적이 해제되지는 않는다. 현재 저장소가 없어 추적 파일 목록이나 git check-ignore 검증은 수행할 수 없다.</p><h2>재현 및 원본 보존</h2><p>source_manifest.csv에 각 원본의 행 수·바이트 수·SHA-256을 기록했고 분석 종료 시 다시 비교했다. 집계 CSV는 파생 결과이며 원본 CSV를 복사하거나 수정하지 않았다. 모든 장비 집계·유형 집계·교차표의 합계가 전체 행 수와 일치하는지 검사했다.</p>')
assert tools.fault_count.sum()==types.fault_count.sum()==cross.values.sum()==len(df)
assert len(gaps)==len(valid)-valid.Tool.nunique()
assert all(hashlib.sha256((ROOT/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in manifest)
style='body{font-family:Malgun Gothic,sans-serif;max-width:1450px;margin:36px auto;color:#172d42;line-height:1.7}table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}th,td{padding:7px 11px;border-bottom:1px solid #dbe3eb;text-align:right}th{background:#eaf1f8}img{max-width:100%}h2{margin-top:40px}'
(OUT/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>PHM 2018 고장 기록 분석</title><style>'+style+'</style>'+''.join(sections)+'</html>',encoding='utf-8')
print(json.dumps(quality,ensure_ascii=False)); print(summary.to_string()); print(types.to_string()); print(cross.to_string())
