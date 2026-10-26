# PHM2018-Predictive-Maintenance

머신러닝 기반 반도체 장비의 이상 징후 분석 및 사전 고장 예측 시스템을 준비하는 Python 개인 프로젝트다.
실제 장비의 시간별 센서 기록을 이용해 운용 중 미래 고장 위험을 예측하는 것이 목표다.
현재 예측 고장 유형, 모델, 예측 선행시간은 **미정**이다. 모델 성능은 아직 산출하지 않았다.

## 데이터셋과 문제 정의

PHM Society 2018 Data Challenge의 Ion Mill Etch Tool 데이터를 사용한다.
[PHM Society 공식 대회 설명](https://phmsociety.org/conference/annual-conference-of-the-phm-society/annual-conference-of-the-prognostics-and-health-management-society-2018-b/phm-data-challenge-6/)은 웨이퍼 제조 공정의 ion mill etch 장비 고장을 미리 예측하는 문제를 설명한다.
원본 재배포 권한·라이선스를 검토하기 전에는 원본을 Git이나 GitHub에 포함하지 않는다.

데이터 루트 아래 `train/`의 장비별 센서 CSV, `train/train_ttf/`의 유형별 잔여시간 라벨,
`train/train_faults/`의 고장 기록을 파일명 장비 ID로 연결한다. `test/`는 이번 분석 범위 밖이다.
센서는 24열(시간·장비·공정 메타데이터 7열, 센서 17열), TTF는 시간과 3개 고장 유형 라벨 4열이다.
고장은 `time`, `fault_name`, `Tool`로 구성된다. 원본은 프로젝트 외부의 기존 위치에 그대로 둔다.

## 완료된 분석과 품질 결과

- 장비 20대, 고장 기록 1,236행. 동일 장비·시간·유형 중복 초과 25행, 중복 제거 조합 1,211개.
- 저압 756행(19대), 고압 340행(19대), 누설 139행(14대), 고압 `[NoWaferID]` 1행.
- 03M02의 저압 기록 301행 집중. 행 수를 독립 고장 사건 수나 장비 고장률로 해석하지 않는다.
- 센서 82,189,440행을 100,000행 청크로 처리. 센서·TTF 시간값과 행 순서 전부 일치.
- 유한 TTF는 같은 장비·유형의 다음 고장까지의 시간 차이와 모두 일치.
- TTF 결측은 이후 동일 유형 고장 기록이 없는 검열 구간과 일치. 입력 결측 센서와 구분한다.
- 센서 결측 1,166셀은 모두 FIXTURESHUTTERPOSITION. 10M01에 1,132셀 집중.
- ROTATIONSPEED는 04M02·10M01에서 장비 내 상수. 동일 시각 인접 간격 117,533개, 긴 관측 공백 존재.
- 고압 사건 4건은 센서 관측 시작 이전(04M02 1건, 09M02 3건).
- 중복 제거 후 사전 창 확보 후보: 저압 292, 고압 82, 누설 73. 3,600 원본 시간 단위 근접 병합 후 273·74·65개.
  이 수는 해당 창의 30행 이상·2,880 이상 관측 span 등 조건을 적용한 잠정 후보다. 독립 수리 사건 수가 아니다.
- 대표 장비 03M02·09M02·06M02의 압력/유량을 고장 전과 비교 운용 구간으로 탐색했다.
  예측 가능한 선행 징후가 있다는 근거는 아직 충분하지 않다.

보고서: [고장 분석](reports/fault_analysis/report.html), [센서 검증](reports/sensor_validation/sensor_validation_report.html).
그래프는 센서 HTML에 직접 포함하고, 별도 SVG도 보존했다. 고장 보고서는 옆의 SVG를 참조한다.
원본 그대로의 보존본은 `reports/archive/`와 `src/legacy/`이며 Git 제외다.
공개 후보 사본은 개인 절대경로를 `${PHM2018_DATA_ROOT}` 등으로 치환했다.
기존 원본 분석 파일은 삭제·수정하지 않았다.

## 폴더

```text
src/          설정, 환경 검사, 기존 분석의 실행 가능한 사본
reports/      경로 정리 보고서와 작은 검토된 집계표
figures/      SVG 시각화 사본
notebooks/    필요 시 추가 (현재 없음)
models/       모델 저장 영역 (현재 모델 없음, 기본 Git 제외)
tests/        설정, 덮어쓰기 방지, 보고서 연결 검증
.local/       개인 설정/검사 로그/마이그레이션 명세 (Git 제외)
.venv/        프로젝트 Python 환경 (Git 제외)
```

## 개발 환경 및 실행

Python 3.12, NumPy 2.3.5, pandas 3.0.1을 기준으로 구성했다. 현재 코드에는 scikit-learn이나 Jupyter가 필요하지 않아 설치하지 않는다.
새 환경에서는 프로젝트 루트 PowerShell에서 다음을 실행한다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

`.env`의 `PHM2018_DATA_ROOT`에 `train/`과 `test/`를 포함한 압축 해제 루트의 로컬 경로를 지정한다.
현재 컴퓨터의 `.env`는 이미 구성되어 있다. 운영체제 환경변수가 `.env`보다 우선한다.
경로를 소스에 직접 적지 않는다. 환경 활성화 없이 아래 명령을 쓸 수 있다.

```powershell
.\.venv\Scripts\python.exe -m src.check_environment
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

환경 검사는 20대 파일의 존재와 작은 헤더만 읽는다. 모델 학습이나 센서 전수 분석을 실행하지 않는다.
분석을 다시 하기로 결정한 뒤에만 아래 모듈을 실행한다. 이번 환경 구성 단계에서는 실행하지 않았다.

```powershell
.\.venv\Scripts\python.exe -m src.analyze_faults
.\.venv\Scripts\python.exe -m src.validate_sensors
```

결과는 기본 `reports/generated/<새 실행 ID>/`에 저장된다. 같은 실행 ID의 기존 폴더는 덮어쓰지 않는다.
검증 모듈은 기존 고장 보고서의 작은 집계 파일을 기준점으로 읽는다.
`PHM2018_OUTPUT_ROOT`와 `PHM2018_RUN_ID`로 출력을 설정할 수 있다.
센서 HTML만 다시 만들 때는 새 검증 출력 또는 임시 사본을 지정한다.

```powershell
$env:PHM2018_SENSOR_REPORT_DIR = 'reports/generated/<실행 ID>/sensor_validation'
.\.venv\Scripts\python.exe -m src.build_sensor_report
Remove-Item Env:PHM2018_SENSOR_REPORT_DIR
```

기존 `reports/sensor_validation` 및 `reports/archive`에서의 보고서 재생성은 차단한다.
전체 센서 검증은 무거운 작업이며 HTML 재생성과 구분해야 한다.
아직 배포·서비스 실행 명령은 없다.

## Git / SourceTree / GitHub

이 폴더를 Git 저장소로 초기화했다. 기존 전역 작성자 설정을 변경하지 않았다.
SourceTree에서는 이 폴더를 기존 로컬 저장소로 추가해 변경 내용을 검토할 수 있다.
현재 **commit 없음, 원격 없음, push 없음**이다. GitHub 연결은 별도 결정 후 진행한다.
파일은 아직 stage하지 않았으므로 `git ls-files`는 비어 있다.
검토 대상 목록과 크기는 Git 제외된 `.local/git_candidate_inventory.csv`에 있다.

```powershell
git status --short --branch
git ls-files --others --exclude-standard
git remote -v
```

`.env`, `.local`, `.venv`, 원본 폴더/파일명, CSV·압축파일·중간 데이터·학습 모델은 기본 제외한다.
작고 검토된 보고서 집계 CSV만 지정 폴더에서 예외로 허용한다. 원본 보존본의 개인 경로도 추적하지 않는다.
새 가공 데이터는 `data/` 또는 `reports/generated/`에서 검토하고, 파일 크기·라이선스·민감 정보·재생성 가능성을 확인한 뒤 추적 여부를 결정한다.
`git add -f`로 제외 규칙을 우회하지 않는다. Notebook 출력도 공개 전 확인한다.

## 향후 계획 (이번 단계에서는 미수행)

1. EDA: 공정 조건·장비 차이·관측 공백과 독립 사건 정의를 추가 검토.
2. 전처리: 중복·결측·운용 구간과 과거 센서 창의 처리 기준 결정.
3. 모델 개발: 고장 유형·선행시간·기준 모델을 근거에 따라 선정.
4. 평가: 시간순·장비·사건 군집 분리, 겹치는 창 제거, 거짓 경보와 선행시간을 포함한 평가 설계.
5. 서비스: 검증된 모델이 있을 때 위험 점수와 경보를 보여주는 로컬 서비스 검토.

## 미검증 사항과 한계

- time의 물리 단위·기준 시각과 센서 스케일·물리 허용 범위는 이 프로젝트의 기존 분석에서 확정하지 않았다.
- 고장 행은 수리 완료·독립 고장 주기를 보장하지 않는다. 근접 병합 기준은 민감도 분석용이다.
- 관측 창의 양끝 조건만으로 내부 공백·독립성·센서 완전성이 보장되지는 않는다.
- 분포/IQR과 대표 시각화는 1/200 계통 표본 기반으로 주기적 공정과의 표본 편향 가능성이 있다.
- 고장 전 비교에는 근접한 이전 고장 직후 데이터가 포함될 수 있다. 비교 운용 구간이 물리적 정상임을 보장하지 않는다.
- 공정 조건 통제와 사건별 재현성이 부족해 선행 징후·성능을 확정하지 않았다. test 데이터와 외부 장비 검증은 미수행.
- TTF 및 미래 고장 정보는 라벨·정합성 검증에만 사용한다. 모델 입력에서 제외한다.
- 향후 정규화·대체·특성 선택은 학습 구간만 사용하고, 시간 순서를 유지하며 사건 군집의 평가 분할 누수를 막아야 한다.
- 센서·TTF 원문은 크기·수정시각 보존을 검사했고 시간열은 정확히 대조했다. 원문 전체 바이트 해시는 따로 계산하지 않았다.
- 기존 HTML의 자동 브라우저 검토는 로컬 URL 정책으로 제한되었다. SVG 렌더링과 HTML 링크·XML 검사를 수행했다.
