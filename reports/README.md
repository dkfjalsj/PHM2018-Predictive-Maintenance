# 분석 결과 보존 정책

- `fault_analysis/report.html`: 고장 기록 분석의 경로 정리본.
- `sensor_validation/sensor_validation_report.html`: 센서·TTF 검증의 경로 정리본.
- `archive/`: 기존 파일의 바이트 동일 사본. 개인 경로와 큰 표본을 포함해 Git 제외.
- `generated/`: 새 실행 결과. 용량·개인 경로 검토 전까지 Git 제외.

공개 후보 보고서는 숫자를 변경하지 않고 로컬 절대경로와 폴더 간 링크만 정리했다.
보고서가 참조하는 SVG도 같은 폴더에 유지해 기존 상대 링크를 보존했다.
소스 경로 `${PHM2018_DATA_ROOT}`는 로컬 설정에서 해석할 위치를 나타낸다.
기존 HTML은 새 프로젝트의 공개 후보 사본이며, 원본 보존본은 archive에 있다.
