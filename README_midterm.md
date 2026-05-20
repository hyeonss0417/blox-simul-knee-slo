# GPU 클러스터 추론 워크로드 스케줄링 최적화 - 중간보고 실험 환경

## 프로젝트 구조

```
blox/
├── simulator_simple.py          # 시뮬레이터 메인 (워크로드 생성 + 가상 클러스터)
├── run_scheduler.py             # [신규] 범용 스케줄러 실행기 (FIFO/LAS/SRTF/SloScoring)
├── las_scheduler.py             # 원본 LAS 스케줄러 실행기
│
├── schedulers/                  # 스케줄링 정책 구현
│   ├── scheduler_policy.py      # 추상 베이스 클래스 (SchedulingPolicy)
│   ├── fifo.py                  # FIFO - 제출 순서대로
│   ├── las.py                   # LAS - 최소 서비스 우선 (Tiresias)
│   ├── srtf.py                  # SRTF - 최단 잔여시간 우선 (Oracle)
│   └── slo_scoring.py           # [신규] 비대칭 SLO 스코어링 스케줄러
│
├── placement/                   # GPU 배치 정책
│   ├── placement_policy.py      # 추상 베이스 클래스
│   ├── consolidated.py          # 통합 배치
│   └── first-gpu.py             # 첫 번째 가용 GPU 배치
│
├── blox/                        # Blox 코어 라이브러리
│   ├── blox_manager.py          # 메인 매니저 (스케줄러↔시뮬레이터 조율)
│   ├── cluster_state.py         # GPU 클러스터 상태 관리
│   ├── job_state.py             # 작업 상태 관리
│   └── deployment/
│       ├── grpc_server_rm.py    # gRPC 서버 (Resource Manager)
│       ├── grpc_proto/          # Protocol Buffer 정의
│       └── grpc_stubs/          # 컴파일된 gRPC 스텁
│
├── workload/                    # 워크로드 생성
│   ├── workload.py              # Workload 클래스 (Philly Trace 파싱)
│   └── parse_philly_jobs.py     # Philly Trace JSON 파서
│
├── generate_synthetic_trace.py  # [신규] 합성 Philly Trace 생성기
├── plot_results.py              # [신규] 실험 결과 시각화 스크립트
├── run_all_baselines.sh         # [신규] 베이스라인 일괄 실행 스크립트
│
├── cluster_job_log              # 합성 트레이스 데이터 (500 jobs)
├── figures/                     # [신규] 실험 결과 그래프
│   ├── avg_jct_comparison.png   # 평균 JCT 비교 (Bar Chart)
│   ├── jct_cdf.png              # JCT 분포 (CDF)
│   └── jct_percentiles.png      # JCT 백분위수 비교
│
└── venv/                        # Python 가상환경
```

## 시뮬레이션 아키텍처

```
┌──────────────────────┐     gRPC      ┌─────────────────────────┐
│  simulator_simple.py │◄────────────►│    run_scheduler.py      │
│                      │              │                          │
│  - 가상 GPU 클러스터  │  GetConfig   │  - 스케줄링 정책 실행     │
│    (32노드 × 4GPU)   │  GetJobs     │  - FIFO/LAS/SRTF/       │
│  - 워크로드 생성      │  Metrics     │    SloScoring 선택       │
│  - JCT 수집          │              │  - 배치 정책 실행         │
└──────────────────────┘              └─────────────────────────┘
```

## 실행 방법

### 1. 환경 설정
```bash
python3 -m venv venv
source venv/bin/activate
pip install grpcio matplotlib pandas grpcio-tools numpy

# gRPC 스텁 컴파일
cd blox/deployment && make grpc && cd ../..
```

### 2. 합성 트레이스 생성
```bash
python generate_synthetic_trace.py
```

### 3. 시뮬레이션 실행 (터미널 2개 필요)
```bash
# 터미널 1: 시뮬레이터
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python simulator_simple.py \
  --cluster-job-log ./cluster_job_log --sim-type trace-synthetic \
  --jobs-per-hour 6 --exp-prefix exp --scheduler Fifo

# 터미널 2: 스케줄러 (Fifo / Las / Srtf / SloScoring)
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python run_scheduler.py \
  --simulate --load 6 --exp-prefix exp --scheduler-name Fifo
```

### 4. 결과 시각화
```bash
python plot_results.py
# 결과: figures/ 디렉토리에 PNG/PDF 생성
```

## 비대칭 스코어링 스케줄러 (SloScoring)

연구제안서 식 (2) 구현:

```
score(slack) = -α × |slack|    if slack < 0   (SLO 위반 임박 → 최우선)
             =  β × slack      if slack ≥ 0   (여유 → 낮은 우선순위)

where slack = SLO_deadline - (current_time + remaining_time)
      α = 10.0, β = 1.0  (α >> β: 위반 페널티가 여유 보상보다 10배)
```

핵심 아이디어: SLO 마감이 가까운 작업은 지수적으로 우선순위가 올라가고, 여유 있는 작업은 선형적으로만 밀려남.

## 실험 결과 요약 (합성 트레이스, Load=6)

| 스케줄러 | Avg JCT (s) | 비고 |
|---------|------------|------|
| FIFO | 68,037 | 제출 순서 기반, Head-of-line blocking |
| LAS | 68,032 | 최소 서비스 우선, Tiresias 방식 |
| SRTF | 68,058 | 최단 잔여시간 우선 (Oracle) |
| SloScoring | 68,062 | 비대칭 스코어링 (프로토타입) |

> 합성 트레이스 + 저부하 환경에서는 정책 간 차이가 미미.
> 실제 Philly Trace와 고부하 환경에서 유의미한 차이 확인 예정.

## 기술적 이슈 및 해결

1. **protobuf 호환성**: Python 3.14 + protobuf 7.x 환경에서 `including_default_value_fields` → `always_print_fields_with_no_presence`로 API 변경
2. **pandas 호환성**: pandas 3.x에서 `DataFrame.append()` 제거됨 → `pd.concat()` 사용
3. **Philly Trace 접근 불가**: GitHub LFS 용량 초과로 원본 trace 다운로드 불가 → 합성 trace 생성으로 대체
4. **gRPC 스텁 미포함**: 레포에 컴파일된 스텁이 없음 → `make grpc`로 수동 컴파일 필요

## 향후 계획

- [ ] 실제 Philly Trace 확보 및 실험
- [ ] 고부하(load=10+) 환경에서 정책 간 차이 분석
- [ ] SloScoring 스케줄러의 α, β 파라미터 튜닝
- [ ] 부하 인식형 배치 정책(Load-Balanced Placement) 구현
- [ ] ML 기반 JCT 예측 모델 통합
