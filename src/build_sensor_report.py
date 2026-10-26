from pathlib import Path
from src.config import PROJECT_ROOT, data_root, output_dir, report_workdir
import pandas as pd
import numpy as np
import html,json
OUT=report_workdir()
eq=pd.read_csv(OUT/'equipment_validation.csv'); col=pd.read_csv(OUT/'column_quality_by_tool.csv'); tt=pd.read_csv(OUT/'ttf_consistency.csv'); events=pd.read_csv(OUT/'event_count_sensitivity.csv'); dist=pd.read_csv(OUT/'sensor_sample_distributions.csv'); mapping=pd.read_csv(OUT/'file_mapping.csv'); ev=pd.read_csv(OUT/'event_coverage.csv')
assert len(eq)==20 and eq.Tool.nunique()==20
assert (eq.rows==eq.ttf_rows).all() and (eq.interval_n==eq.rows-1).all()
col['invalid_numeric']=col.rows-col.finite-col.missing-col.nonfinite
eq['distinct_times_if_sorted']=np.where(eq.backward_intervals.eq(0),eq.rows-eq.zero_intervals,np.nan)
eq.to_csv(OUT/'equipment_validation.csv',index=False,encoding='utf-8-sig')
col.to_csv(OUT/'column_quality_by_tool.csv',index=False,encoding='utf-8-sig')
def table(d):return d.to_html(index=False,border=0,float_format=lambda x:f'{x:.3e}' if 0<abs(x)<.0001 else f'{x:,.4f}',na_rep='해당 없음',escape=True)
def graphic(title,body,h=420):return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1400 {h}" role="img" aria-label="{html.escape(title)}"><rect width="1400" height="{h}" fill="white"/><g font-family="Malgun Gothic,sans-serif" fill="#172d42"><text x="25" y="30" font-size="21">{html.escape(title)}</text>{body}</g></svg>'
def bars(df,value,title):
    mx=df[value].max();body=''
    for i,(_,r) in enumerate(df.iterrows()):
        y=65+28*i;w=1100*r[value]/mx
        body+=f'<text x="10" y="{y+17}" font-size="13">{r.Tool}</text><rect x="100" y="{y}" width="{w}" height="21" fill="#2563a6"/><text x="{w+110}" y="{y+17}" font-size="12">{r[value]:,.0f}</text>'
    return graphic(title,body,660)
parts=['<h1>PHM 2018 센서·TTF 품질 및 정합성 검증</h1><p>범위: 학습용 센서·TTF·고장 기록의 품질, 시간 정합성, 사건 단위 적합성 및 탐색적 시각화. 모델 학습과 최종 예측 대상 선정은 수행하지 않았다.</p>']
parts+=['<p><a href="../fault_analysis/report.html">기존 고장 기록 보고서 열기</a></p>']
parts+=['<h2>핵심 결과</h2>',f'<p>장비 {len(eq)}대, 센서 {int(eq.rows.sum()):,}행, TTF {int(eq.ttf_rows.sum()):,}행. 시간 행 불일치 합계 {int(eq.time_row_mismatch.sum()):,}개. 시간열의 장비별 순서와 SHA-256 해시까지 일치한 장비 {int(eq.time_sequence_hash_equal.sum())}대.</p>',f'<p>TTF에서 다음 동일 유형 고장까지의 차이와 일치하지 않는 유한값 {int(tt.mismatch_ge.sum()):,}개, 미래 동일 유형 고장이 있는데 TTF가 결측/비유한인 셀 {int(tt.missing_with_future.sum()):,}개, 마지막 동일 유형 고장 이후 유한 TTF {int(tt.unexpected_finite_after_last.sum()):,}개. 아래 세부 표는 정상적인 우측 검열과 정합성 문제를 구분한다.</p>']
parts += [f'<p>17개 센서에서 결측 총 {int(col.loc[col.role.eq("sensor"),"missing"].sum()):,}셀, 무한대·숫자 변환 실패 0셀. 결측은 모두 FIXTURESHUTTERPOSITION에 있고 10M01에 1,132셀이 집중된다. ROTATIONSPEED는 04M02와 10M01에서 장비 내 상수다. 고압 사건 4건은 센서 관측 시작 이전이다. 아래 사건 수는 관측 창과 근접 병합 기준을 적용한 잠정 후보이며 독립 수리 사건으로 확정한 수가 아니다.</p>']
parts+=['<h2>기존 분석 및 파일 매칭</h2><p>기존 고장 분석의 20대·1,236행·중복 초과 25행을 다시 대조했다. 기존 report.html과 파생 파일은 SHA-256 비교로 보존을 확인했다. 원본은 ${PHM2018_DATA_ROOT}\\train 아래에 있다. 센서와 TTF 파일명은 동일하며, 고장 파일명 접두의 장비 ID와 대응한다. 센서 Tool 열은 파일명 ID와 모든 행에서 대조했다. TTF에는 Tool 열이 없으므로 파일명으로 연결한다.</p>',table(mapping)]
parts+=['<p>장비 파일 20쌍의 센서는 모두 동일한 24개 열, TTF는 동일한 4개 열이다. train_ttf에서 장비 파일명 패턴 밖의 zNex~\u0024hare01_M01_DC_train.csv (668바이트) 1개가 추가로 발견되어 장비 매칭에서 제외했다. 해당 파일은 분석 대상 CSV 스키마를 따르지 않는 응용프로그램 잠금 정보로 보이는 부가 파일이며 원본에서 삭제하지 않았다.</p>']
parts+=['<h2>처리 및 계산 방법</h2><p>센서·TTF는 pandas chunksize=100,000으로 순차 처리했다. 전체 센서 테이블을 한 번에 메모리에 올리지 않았다. 행 수, 결측 수, 최솟값·최댓값, 시간 역행·중복 및 TTF 연결은 전수 검사다. 측정 간격은 파일 순서를 유지한 인접 time 차이이며 청크 경계를 포함한다. 간격별 빈도표를 누적해 중앙값·95백분위수를 정확히 계산했다. 숫자 time은 단위·기준 날짜가 제공되지 않아 원본 시간 단위로만 표시한다.</p><p>센서 열은 FLOWCOOLPRESSURE 등을 포함하는 원본 8번째 이후 열이다. time, Tool, stage, Lot, runnum, recipe, recipe_step은 메타데이터로 구분했다. 결측률은 빈/NA 셀 수÷해당 열 전체 행 수×100, 장비 센서 결측률은 센서 결측 셀÷(행 수×센서 열 수)×100이다. 숫자형 무한대는 결측과 따로 집계했다. 상수 여부는 모든 유한값의 min=max로 판정하고 유효값 0개는 상수로 분류하지 않았다.</p><p>분포와 시각화에는 장비별 원본 행 번호 0,200,400,…의 결정적 1/200 계통 표본을 썼다. 분위수와 IQR 이상치 비율은 이 표본의 추정치이며 전수 비율이 아니다. 주기적 공정과 표본 간격이 겹치는 편향 가능성이 있다. IQR 경계는 Q1−1.5×IQR, Q3+1.5×IQR이며 경계 밖 값을 물리적 센서 오류로 단정하지 않았다. 스케일·물리 단위·공정 허용 범위가 확인되지 않아 음수 자체도 오류로 취급하지 않았다. 이산 센서나 IQR=0인 열에서는 허용된 상태 변화도 IQR 경계 밖으로 잡힐 수 있다.</p>']
parts+=['<h2>장비별 측정 기록과 시간 정합성</h2>',bars(eq,'rows','장비별 센서 측정 행 수'),table(eq)]
parts += [f'<p>동일 시각의 연속 측정 간격은 총 {int(eq.zero_intervals.sum()):,}개다. 시간 역행이 없는 장비의 서로 다른 시각 수는 행 수−0 간격 수다. 동일 시각 여러 행을 유지한 채 TTF와 비교했으며, 이 행들이 센서 전체 열까지 동일한 중복인지 여부는 이번 검사로 판정하지 않았다. 가장 긴 측정 공백은 {eq.interval_max.max():,.0f} 원본 시간 단위다. 긴 공백을 고장 전 연속 관측으로 간주하지 않도록 사건 수 기준과 함께 확인해야 한다.</p>']
parts+=['<h2>열별 품질</h2><p>아래 장비별 표에서 dtype은 청크별 pandas 추론 타입의 합집합이다. Tool은 식별자 문자열로 취급하고, 실제 관측 ID는 위 equipment_validation의 tool_mismatch로 검사했다. 결측이 있는 정수형 열은 float64로 읽힐 수 있으므로 타입 이름만으로 물리 의미를 판정하지 않는다.</p>']
agg=col[col.role.eq('sensor')].groupby('column').agg(rows=('rows','sum'),missing=('missing','sum'),nonfinite=('nonfinite','sum'),minimum=('minimum','min'),maximum=('maximum','max'),constant_tools=('constant_finite','sum')).reset_index();agg['missing_pct']=agg.missing/agg.rows*100
agg.to_csv(OUT/'sensor_column_summary.csv',index=False,encoding='utf-8-sig')
parts+=[table(agg),'<h3>표본 분포 및 IQR 경계 밖 값</h3>',table(dist),'<details><summary>장비·컬럼별 dtype / 결측 / 상수 상세</summary>',table(col),'</details>']
parts+=['<h2>TTF와 고장 시각 연결</h2><p>유형 c, 시각 t에 대해 f*=min{고장시각 f : f≥t, 같은 장비·같은 원문 fault_name}를 구하고 TTF_c(t)=f*−t를 모든 행에서 비교했다. 동일 시각 중복은 고장시각 집합에서 제거했다. 엄격한 미래 f&gt;t의 일치 수도 별도 기록했다. 센서가 고장시각과 정확히 같을 때의 0 TTF 허용 여부는 이 두 열로 구분한다. TTF의 결측과 미래 고장 부재가 동시에 발생하면 마지막 사건 이후 우측 검열로 해석할 수 있다. 세 TTF 유형에 없는 [NoWaferID]는 일반 고압으로 병합하지 않았으며 TTF 직접 검증 대상에서 제외했다.</p>',table(tt)]
ttsum=tt.groupby('column').sum(numeric_only=True).reset_index();ttsum['missing_pct']=100*ttsum.missing/ttsum.rows
parts += ['<h3>TTF 유형별 전수 합계</h3>',table(ttsum),'<p>전수 검증에서 유한 TTF는 모두 f≥t 규칙과 정확히 일치했다. 0 TTF도 해당 유형의 실제 고장시각과 일치했다. 모든 TTF 결측은 이후 동일 유형 고장이 없는 경우에만 발생했으므로, 이 데이터 범위에서는 라벨 불일치가 아닌 검열로 구분했다. 고장 자체가 기록되지 않은 장비·유형의 전 구간 결측도 여기에 포함된다.</p>']
parts+=['<h2>독립 사건 후보와 관측 가능성</h2><p>고장 한 행을 독립 사건으로 간주하지 않았다. 먼저 (Tool,time,fault_name) 중복을 제거했다. 그 뒤 같은 장비·유형에서 인접 고장 간격이 임계값 이하인 기록을 연결해 하나의 군집으로 묶었다. 연결 군집은 전체 길이가 임계값보다 길어질 수 있다. 군집 대표는 첫 기록이며 이후 사건으로 바꿔 앞선 관측 부족을 보완하지 않았다. 임계값 0·60·600·3,600은 복구 이력 없이 정한 민감도 분석용 값이다.</p><p>학습 가능 후보 수(eligible_cluster_anchors)는 대표시각이 센서 min~max 안에 있고, [대표시각−3,600, 대표시각) 구간 시작도 관측 범위 안이며, 그 구간에 센서 30행 이상·첫/끝 센서 간격 2,880 이상이 있는 군집 수다. 이 기준은 시간창의 양끝 관측을 점검하지만 내부 대규모 결측/정지·센서별 완전성·독립 수리 주기는 보장하지 않는다. 따라서 실제 학습 가능한 사건 수의 잠정 상한 후보이며 확정된 독립 사건 수가 아니다. 유형 간 가까운 고장은 별개 군집으로 남아 있으므로 유형별 후보 수 합계도 독립 사건 수가 아니다.</p>',table(events)]
outside=ev[~ev.in_range].merge(eq[['Tool','first_time','last_time']],on='Tool',how='left')
parts += ['<h3>관측 범위 밖 사건</h3>',table(outside),'<p>4건 모두 관측 시작 이전이다. 04M02 고압 1건, 09M02 고압 3건이며, 중복 제거 후 사건 중에서 구분했다. 이들은 해당 파일의 센서 관측으로 사전 예측 창을 구성할 수 없다.</p>']
parts+=['<h2>대표 장비의 고장 전 구간과 일반 운용 구간</h2><p>장비는 원본 고장 기록 수만으로 선정했다. 03M02는 저압 최다 301건, 09M02는 고압 최다 62건, 06M02는 누설 최다 28건이다. 센서 모양이나 효과 크기로 선정하지 않았다. 시각화의 고장은 유형을 합친 서로 다른 고장시각이다. 고장 전은 다음 고장까지 (0,3,600]인 행, 비교 구간은 이전·다음 고장 모두에서 3,600 초과 떨어진 행이다. 비교 구간은 고장이 기록되지 않은 운용 구간일 뿐 정상 상태가 보장되지는 않는다. 미래 사건 정보는 사후 구간 정의와 라벨 검증에만 사용했다.</p>']
parts+=['<p>근접 고장이 있는 경우 고장 전 구간에 이전 고장 시각 또는 직후의 측정이 포함될 수 있다. 수리 완료 시각이 없어 이를 독립적인 복구 후 구간으로 분리할 수 없다. 그림은 원시 고장시각에 정렬한 탐색용 비교이며, 이러한 기록을 사전 예측 성공의 증거로 사용하지 않았다.</p>']
compare=[];strata=[]
for tool in ['03M02','09M02','06M02']:
    pre=pd.read_csv(OUT/(tool+'_pre_window_sample.csv')); general=pd.read_csv(OUT/(tool+'_general_sample.csv'))
    parts+=[f'<h3>{tool}</h3>',f'<p>표본 행 수: 고장 전 {len(pre):,}, 비교 운용 {len(general):,}. 행 단위 비교는 자기상관과 사건별 가중 차이가 있으므로 독립 검정으로 사용하지 않았다.</p>']
    for sensor in ['FLOWCOOLPRESSURE','FLOWCOOLFLOWRATE']:
        pa=pd.to_numeric(pre[sensor],errors='coerce').dropna(); ga=pd.to_numeric(general[sensor],errors='coerce').dropna()
        if not len(pa) or not len(ga): parts+=['<p>유효 표본 부족으로 그래프 생략.</p>'];continue
        r=dict(Tool=tool,column=sensor,pre_n=len(pa),general_n=len(ga),pre_median=pa.median(),general_median=ga.median(),median_difference=pa.median()-ga.median(),pre_q10=pa.quantile(.1),pre_q90=pa.quantile(.9),general_q10=ga.quantile(.1),general_q90=ga.quantile(.9));compare.append(r)
        low=min(pa.quantile(.01),ga.quantile(.01)); high=max(pa.quantile(.99),ga.quantile(.99));span=high-low or 1
        def sy(v):return 340-260*(v-low)/span
        body=f'<text x="85" y="65" font-size="13">縦축: 관측값, 표시범위는 두 구간의 1~99백분위</text>'
        body=body.replace('縦축','세로축')
        for v in np.linspace(low,high,5):body+=f'<line x1="100" y1="{sy(v)}" x2="1250" y2="{sy(v)}" stroke="#ddd"/><text x="10" y="{sy(v)+4}" font-size="12">{v:.3g}</text>'
        generalmedian=ga.median();body+=f'<line x1="100" y1="{sy(generalmedian)}" x2="1250" y2="{sy(generalmedian)}" stroke="#d16b28" stroke-dasharray="7 4"/>'
        for k in range(12):
            a=pre.loc[(pre.relative_time>=-3600+k*300)&(pre.relative_time<-3600+(k+1)*300),sensor].dropna()
            if len(a):
                x=140+96*k;q10,q50,q90=a.quantile([.1,.5,.9]); y1=sy(min(max(q10,low),high));y2=sy(min(max(q90,low),high));ym=sy(min(max(q50,low),high))
                body+=f'<line x1="{x}" x2="{x}" y1="{y1}" y2="{y2}" stroke="#2563a6" stroke-width="5" opacity=".5"/><circle cx="{x}" cy="{ym}" r="5" fill="#2563a6"/>'
                body+=f'<text x="{x-25}" y="365" font-size="11">{-3600+k*300}</text>'
        body+='<text x="100" y="395" font-size="13">고장까지 상대 시간 (원본 단위). 파랑: 300단위 구간 중앙값·10~90백분위, 주황 점선: 비교 운용 중앙값</text>'
        svg=graphic(f'{tool} {sensor}: 고장 전 추이와 비교 운용',body);name=tool+'_'+sensor+'_trajectory.svg';(OUT/name).write_text(svg,encoding='utf-8');parts.append(svg)
        # Full comparison as normalized histogram; tails clipped for display only.
        edges=np.linspace(low,high,31);hp,_=np.histogram(pa,bins=edges);hg,_=np.histogram(ga,bins=edges);hp=hp/len(pa);hg=hg/len(ga);ymax=max(hp.max(),hg.max()) or 1;body=''
        for j,(a,b) in enumerate(zip(hp,hg)):
            x=100+j*38;body+=f'<rect x="{x}" y="{340-250*a/ymax}" width="17" height="{250*a/ymax}" fill="#2563a6"/><rect x="{x+18}" y="{340-250*b/ymax}" width="17" height="{250*b/ymax}" fill="#d16b28"/>'
        for value in np.linspace(0,ymax,4):
            y=340-250*value/ymax;body+=f'<text x="25" y="{y+4}" font-size="12">{value:.3f}</text>'
        body+=f'<text x="100" y="370">{low:.4g}</text><text x="1200" y="370">{high:.4g}</text><text x="100" y="400">파랑: 고장 전 / 주황: 비교 운용. 세로축: 전체 구간 표본 대비 bin 비율, 양끝 1% 부근 꼬리는 표시 생략.</text>'
        svg=graphic(f'{tool} {sensor}: 구간별 표본 분포',body);(OUT/(tool+'_'+sensor+'_distribution.svg')).write_text(svg,encoding='utf-8');parts.append(svg)
    # Compare within the same observed recipe/stage/step, descriptive only.
    keys=['stage','recipe','recipe_step']
    a=pre.groupby(keys,dropna=False).FLOWCOOLPRESSURE.agg(pre_n='count',pre_median='median');b=general.groupby(keys,dropna=False).FLOWCOOLPRESSURE.agg(general_n='count',general_median='median');m=a.join(b,how='inner').reset_index();m=m[(m.pre_n>=30)&(m.general_n>=30)].copy();m['Tool']=tool;m['difference']=m.pre_median-m.general_median;strata.append(m)
comparison=pd.DataFrame(compare);comparison.to_csv(OUT/'representative_window_comparison.csv',index=False,encoding='utf-8-sig');strat=pd.concat(strata,ignore_index=True);strat.to_csv(OUT/'pressure_comparison_by_recipe_stage.csv',index=False,encoding='utf-8-sig')
parts += ['<p>압력 중앙값 차이(고장 전−비교 운용)는 03M02 +0.01195, 09M02 −0.00998, 06M02 +0.02318로 장비 간 방향이 일정하지 않았다. 06M02에서는 고장 전 하위 꼬리의 차이가 보이나 회복 상태와 공정 전환의 영향이 남아 있다. 동일 공정 조합에서 양쪽 표본 30개 기준을 충족한 것은 03M02의 1개 조합뿐이고, 09M02와 06M02는 없었다. 따라서 공정 조건을 충분히 통제한 반복 증거로 보기 어렵다.</p>']
parts+=['<h3>구간별 수치 비교</h3>',table(comparison),'<h3>동일 stage·recipe·recipe_step 내 압력 비교</h3><p>두 구간에 유효 압력 표본이 각각 30개 이상 있는 공정 조합만 포함했다. 이 조정만으로 Lot·runnum·장비 상태·시간 추세의 교란을 제거할 수는 없다.</p>',table(strat),'<h2>예측 징후에 대한 판단과 누수 방지</h2><p><b>현재 결과만으로 재현 가능한 사전 예측 징후가 있다고 확정할 근거는 충분하지 않다.</b> 고장 전/운용 분포의 차이와 시간별 중앙값은 탐색 결과이며 공정 조건·사건 집중·반복 기록·가동 정지·표본 추출의 영향을 받는다. 고장 직전 값이 다르더라도 충분한 선행시간의 예측 가능성, 미관측 사건·장비에 대한 재현성 및 거짓 경보율을 입증하지 않는다. 특히 현재 그림은 가까운 다음 고장에 정렬된 행 단위 통계로 사건별 동일 가중 통계가 아니다.</p><p>이번 단계에서 입력변수나 모델을 만들지 않았다. train_ttf, time+TTF로 복원되는 미래 고장시각, 미래 사건 거리, 사후 구간 구분은 모델 입력에서 배제해야 한다. 향후 검증에서는 시간순서와 사건 군집 단위 분리를 유지하고 겹치는 고장 전 창이 학습·평가 양쪽으로 갈라지지 않게 해야 한다. 임계값·분포 경계·정규화·결측 대체를 전체 기간에서 배운 뒤 평가에 쓰면 누수가 되므로 향후에는 학습 구간만으로 계산해야 한다. 이번 전체기간 IQR은 품질 진단용이며 학습용 변환이 아니다. 정상/음성 구간 선택에도 미래 고장을 이용했으므로 해당 비교를 실시간 배포 규칙이나 성능 평가로 사용할 수 없다.</p>']
parts+=['<h2>보존과 재현</h2><p>센서/TTF는 읽기 전후 파일 크기와 수정시각을 확인했고, 작은 고장 CSV는 SHA-256도 대조했다. 대용량 센서 원문 전체의 바이트 해시는 별도 계산하지 않았으며 시간열 해시와 행별 비교를 수행했다. 기존 보고서와 결과 파일은 바이트 해시로 보존 확인했다. 기존 .gitignore의 train/ 및 원본 폴더 제외 규칙은 유효한 패턴이며 원본을 프로젝트에 복사하지 않았다. 현재 폴더는 Git 저장소가 아니므로 원격 추적 상태를 검사할 수 없다.</p><p>아래 CSV는 이번 결과 폴더의 파생 집계다. 전수 통계와 표본 통계를 파일명·본문에서 구분했다. 수행 스크립트: validate_sensors.py, build_sensor_report.py. 기존 보고서의 브라우저 및 그래프 검토 이력은 reports/archive에 보존되어 있다. 재생성한 보고서는 별도로 검토해야 한다. 그래프는 외부 이미지 파일에 의존하지 않도록 HTML에 직접 포함했다. 이 단계에서 분석을 종료한다.</p>']
for name in ['equipment_validation.csv','column_quality_by_tool.csv','sensor_column_summary.csv','sensor_sample_distributions.csv','ttf_consistency.csv','file_mapping.csv','event_coverage.csv','event_count_sensitivity.csv','representative_window_comparison.csv','pressure_comparison_by_recipe_stage.csv','source_metadata_manifest.csv']:
    parts.append(f'<p><a href="{name}">{name}</a></p>')
style='body{font-family:Malgun Gothic,sans-serif;max-width:1450px;margin:36px auto;padding:0 24px;color:#172d42;line-height:1.7}table{border-collapse:collapse;font-size:12px;display:block;overflow:auto;margin:18px 0}th,td{padding:7px 10px;border-bottom:1px solid #dbe3eb;text-align:right;white-space:nowrap}th{background:#eaf1f8}svg{width:100%;height:auto;margin:16px 0}h2{margin-top:42px}details{margin:20px 0}'
target=OUT/'sensor_validation_report.html';target.write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>PHM 2018 센서 검증</title><style>'+style+'</style>'+''.join(parts)+'</html>',encoding='utf-8')
print(eq[['Tool','rows','time_row_mismatch','sensor_missing_pct']].to_string(index=False));print(tt.groupby('column').sum(numeric_only=True).to_string());print(events.to_string(index=False));print(comparison.to_string(index=False));print('REPORT',target.resolve())
