# Knee-SLO: Non-Oracle SLO-Aware Scheduling for GPU Inference

연세대 26-1 컴퓨터종합설계 캡스톤 — 팀 **젠슨황팀** (윤영준, 박준열, 전현성, 부광민)

기반: [Blox simulator (EuroSys '24)](https://github.com/ericjypark/blox-simul)
워크로드: Alibaba 2026 GenAI inference trace (Stable Diffusion serving)

## 📄 결과 보고서

- **HTML (권장)**: [`docs/report_v2/report.html`](docs/report_v2/report.html) — 사이드바 TOC + 임베드 차트
- Markdown: [`docs/report_v2/report.md`](docs/report_v2/report.md)
- 차트 (PNG / PDF): [`docs/report_v2/figures/`](docs/report_v2/figures/)
- 스타일시트: `docs/report_v2/style.css`

## 🎯 핵심 발견

| 시나리오 | 결과 | 본문 |
| -------- | ---- | ---- |
| Closed batch 추론 (4 GPU, 200/hr) | ⭐ **MetaSrtf (non-Oracle) = oracle SRTF: 40.1s** | §9bis |
| Open over-load (16 GPU, 4000/hr) | 🛡️ bucket 변형이 LAS/SRTF starvation 회피 | §9ter |
| Metadata predictor | R² 0.394 (category-mean baseline 대비 26% 더 정확) | §9quater |
| 합성 training-like | LAS 가 Knee 모든 변형 압도 (negative) | §4–§8 |
| 단일-pool 추론 | 모든 17 개 알고리즘 동등 (under-saturated) | §9 |

## 🧠 알고리즘

- `schedulers/knee_slo.py` — 메인 알고리즘 (3-zone urgency × bucket × LAS hybrid)
- `schedulers/knee_slo_{nonpreempt,class,classdur,adaptive}.py` — 변형
- `schedulers/{las_slo,srtf_slo,meta_pred}.py` — bucket-based + metadata-pred 변형
- `schedulers/{sjf,edf,llf,hrrn}.py` — 추가 baselines

## 🚀 재현

```bash
# 1. 환경 준비
source venv/bin/activate

# 2. Metadata predictor 학습 (offline, ~1s)
python build_metadata_predictor.py

# 3. 단일 실험 (예: closed batch MetaSrtf)
BLOX_NUM_MACHINES=2 BLOX_GPUS_PER_MACHINE=2 \
    SCHED=MetaSrtf EXP_PREFIX=test PORT_BASE=50050 \
    LOAD=200 START=10 STOP=60 ROUND_DURATION=10 \
    bash run_one_experiment.sh

# 4. 그리드 (Wave 1~5)
bash run_all_waves.sh

# 5. 보고서 갱신
python generate_summary.py
python compile_final_report.py
python build_html.py
```

자세한 재현 방법: 보고서 부록 A.

## 📊 파일 구조

```
schedulers/             스케줄러 구현
placement/              GPU placement 정책
workload/               Workload 생성 (Alibaba trace 변환 포함)
blox/                   Blox 시뮬레이터 코어 + gRPC stubs
simulator_simple.py     시뮬레이터 진입점
run_scheduler.py        스케줄러 진입점
run_one_experiment.sh   단일 실험 launcher

# 그리드 / 분석
run_grid.sh             Wave 1: Knee 하이퍼파라미터 그리드
run_grid_wave2.sh       Wave 2: 알고리즘 확장
run_grid_wave3.sh       Wave 3: 부하 민감도
run_grid_wave4.sh       Wave 4: 극단 / 조합 변형
run_grid_wave5.sh       Wave 5: SLO 24h 재캘리브레이션
run_grid_inference.sh   Wave R: 실제 추론 워크로드
run_meta_win.sh         Meta-Win: 16 GPU stability test
run_all_waves.sh        Master orchestrator

# 시각화 / 리포트
plot_v2_results.py      알고리즘 비교 그래프
plot_w3_loadsweep.py    부하 민감도 곡선
plot_slo_curves.py      다중 SLO target curve
plot_inference.py       추론 결과 시각화
plot_algorithm.py       알고리즘 동작 시각화 (urgency, zones 등)
generate_summary.py     보고서 자동 표/findings 갱신
compile_final_report.py 최종 exec summary 합성
build_html.py           HTML 변환

# 데이터
cluster_job_log         Alibaba 2026 GenAI trace (변환된 Philly JSON)
metadata_pred.json      훈련된 prediction 결과 (lookup table)
docs/report_v2/         최종 보고서 폴더
  ├── report.md         원본 마크다운
  ├── report.html       사이드바 TOC + 임베드 차트
  ├── style.css         report HTML 스타일
  └── figures/          모든 PNG / PDF 차트
```

## 📝 라이센스 & 인용

원본 Blox는 MIT 라이센스. 본 fork의 추가 코드는 같은 라이센스를 따른다.

원본 README는 [`README_midterm.md`](README_midterm.md) 참조.
