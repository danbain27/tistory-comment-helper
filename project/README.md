# Bybit 15m LONG-only 전략 연구 시스템

PCA 시장상태 필터 + RSI/BB/Volume 진입 + ATR 그리드(물타기) + 평균단가 TP 를
**과거 데이터 분석 → 백테스트 → 파라미터 최적화 → Out-of-Sample / Walk-forward 검증 →
Monte Carlo** 순으로 검증하는 연구 파이프라인입니다.

목적은 "수익 나는 백테스트"를 만드는 것이 아니라, **이 전략을 채택/폐기할 근거를 숫자로
남기는 것**입니다. 파이프라인은 성능이 안 나오면 `폐기` 판정을 그대로 출력합니다.

---

## 1. 설치 & 데이터 준비

```bash
pip install -r requirements.txt

# 상장된 USDT 무기한 전 종목 다운로드 (유동성/상장기간 필터 통과분)
python fetch_data.py --universe all --days 1095 --workers 4

# 거래대금 상위 100개만
python fetch_data.py --universe top100 --days 1095

# 특정 종목만
python fetch_data.py --symbols BTCUSDT ETHUSDT --funding
```

- 기본 필터: 24h 거래대금 ≥ 500만 USDT, 상장 ≥ 365일 (`--min-turnover`, `--min-listed-days`)
- 이미 받은 CSV가 있으면 **마지막 봉 이후만 이어서** 받습니다 (`--force` 로 전체 재수집)
- 통과 종목 목록은 `data/universe.csv` 에 저장됩니다
- 규모 감각: 약 400종목 × 3년 15분봉 ≈ 4,200만 봉 / 디스크 2~3GB / 다운로드 1~3시간
  (짧게 보려면 `--days 365` 또는 `--universe top100`)

## 2. 실행

```bash
# 받아둔 전 종목 스캔 -> 상위 바스켓 심층 연구 -> 유니버스 전체 재검증
python -m src.run_research --universe local --jobs 8

# 데이터도 그 자리에서 받으며 상장 전 종목 스캔
python -m src.run_research --universe all --jobs 8

# 상위 100개만
python -m src.run_research --universe top100 --jobs 8

# 특정 종목 (기존 방식)
python -m src.run_research --symbols BTCUSDT ETHUSDT SOLUSDT

# 빠른 확인 / 데이터 없는 환경 점검
python -m src.run_research --universe local --quick --jobs 8
python -m src.run_research --universe all --allow-synthetic --synthetic-symbols 14 --quick
```

주요 옵션: `--basket-size 8` (심층 연구 종목 수), `--min-bars 20000` (히스토리 부족 종목 제외),
`--jobs` (스캔/검증 병렬), `--skip-scan` (스캔 없이 바로 심층 연구).

무결성 테스트: `python -m tests.test_engine`

## 3. 파이프라인 구조 (전 종목 스캔)

```
STEP 0   유니버스 확정        상장 USDT 무기한 전 종목 -> 유동성/상장기간/히스토리 필터
STEP 0b  스트리밍 스캔         종목당 1개씩 로드 -> 지표/PCA/기준전략 평가 -> 즉시 해제
                              (train + validation 만 사용, test 는 손대지 않음)
         └ 유니버스 통합 통계   조건별 forward return 을 전 종목 합산 (표본 수백만 봉)
         └ PCA loading 안정성  PC 의미가 종목 간 일치하는지 검증
         └ 바스켓 선정         screen_score = train/val 일관성 가중 상위 N 종목
STEP 1~12  바스켓 심층 연구    진입조건/PCA필터/ATR그리드/TP·SL 탐색 (validation only)
STEP 13    walk-forward       fold마다 PCA·파라미터 재적합
STEP 13b   유니버스 재검증     확정된 설정을 전 종목 test 구간에 적용
                              (선정에 관여하지 않은 종목 포함 = 진짜 OOS)
STEP 14~16 MC / 스트레스 / 국면 분석 / 11개 항목 자동 판정
```

**메모리**: 스캔은 종목을 하나씩 처리하고 버리므로 종목 수와 무관하게 일정합니다
(워커당 ~100MB). 심층 연구 구간만 바스켓(기본 8종목)을 메모리에 올립니다.

**소요 시간(참고)**: 400종목 스캔 `--jobs 8` 기준 5~15분, 심층 연구 5~20분,
유니버스 재검증 3~10분.

**생존 편향 경고**: Bybit 상장 목록에는 **상장폐지된 코인이 없습니다.** 폭락 후 사라진
종목이 통계에서 빠지므로 유니버스 전체 수치는 실제보다 좋게 나옵니다. 리포트에도 이
경고가 함께 출력됩니다.

## 3b. 출력물

```
results/
  universe.csv               스캔 대상 종목과 거래대금/상장일
  universe_scan.csv          종목별 스캔 결과 + screen_score (바스켓 선정 근거)
  pooled_feature_analysis.csv 전 종목 합산 조건별 forward return (t값 포함)
  pca_loading_stability.csv  PC loading 의 종목 간 평균/표준편차
  universe_validation.csv    확정 설정을 전 종목 test 구간에 적용한 결과
  data_quality.csv           결측/중복/갭/이상치 점검
  pca_components.csv         각 PC의 설명력과 top loading 해석
  feature_analysis.csv       바스켓 종목별 구간 통계
  entry_conditions.csv       개별 조건의 forward-return edge
  parameter_test.csv         entry / PCA / grid / TP·SL 탐색 전체 결과
  parameter_sensitivity.csv  파라미터 값별 점수 분포 (과최적화 진단)
  filter_stack.csv           A(RSI) -> F(+ADX) 누적 효과 = PCA 기여도 검증
  backtest_results.csv       train / validation / test 성과
  walkforward.csv            fold별 val -> test 성과와 선택된 파라미터
  walkforward_summary.csv    종목별 walk-forward OOS 집계
  monte_carlo.csv            거래 재표본 시뮬레이션
  cost_stress.csv            수수료 2배 / 슬리피지 / 진입지연 스트레스
  regime_analysis.csv        시장 국면별 성과
  trades.csv                 OOS 거래 로그
  REPORT.md                  사람이 읽는 최종 리포트 + 결론
charts/                      equity, drawdown, monthly, trade dist, PCA, sensitivity
strategy_config.yaml         실봇에 넣을 최종 규칙 + 성과 스냅샷
```

## 4. 과최적화 방지 장치 (설계상 지켜지는 것)

| 항목 | 구현 |
|---|---|
| 미래 데이터 참조 | 신호는 봉 t 종가에서 계산, 체결은 t+1 **시가** (`entry_delay_bars=1`) |
| 진입봉 내 청산 | `exit_from_next_bar=True` — 진입한 봉에서는 TP/SL/그리드 체결 없음 |
| 봉 내부 처리 순서 | 펀딩 → 청산 → SL → TP(그리드 체결 **전** 평균단가 기준) → 그리드 → 최대보유 (의도적으로 불리하게) |
| PCA 데이터 누수 | Scaler/PCA는 **train 구간만** fit, val/test는 transform. walk-forward는 fold마다 재적합 |
| PC 부호 안정화 | 최대 |loading| feature의 부호를 +로 고정 → 재적합해도 필터 방향이 뒤집히지 않음 |
| PC 임계값 | 절대값이 아니라 **train 분포 분위수**로 지정 → 재적합 시 스케일 드리프트에 강함 |
| 파라미터 선택 | validation에서만 선택, test는 **측정 전용** |
| 단일 최고값 선택 방지 | `plateau_pick` — 파라미터 공간에서 **이웃 조합 평균 점수**가 가장 높은 지점을 선택 |
| 선택 기준 | ROI가 아니라 `robust_score` = CAGR/MDD + Sharpe + PF, 표본 부족·청산 발생 시 실격 |
| 비용 | taker 0.055% / maker 0.02% / 슬리피지 / 8시간 펀딩비 반영, 2배 스트레스 테스트 |
| 포지션 한도 | 전 그리드 체결 시 총 명목가 = 자기자본 × 레버리지 (초과 불가), 최대 그리드 횟수·최대 보유시간 제한 |
| 청산 모델링 | isolated margin 근사 청산가 계산, 청산 발생 전략은 자동 실격 |

`tests/test_engine.py` 는 이 중 핵심을 **실제로 검증**합니다.
특히 `test_future_mutation_does_not_change_past` 는 데이터 뒷부분을 무작위로 훼손해도
그 이전에 종료된 거래가 한 건도 바뀌지 않음을 확인합니다 (look-ahead의 강한 반증).

## 5. 최종 판정 기준

`run_research.py` STEP 16 이 11개 항목을 자동 채점합니다.

1. OOS 수익(심볼 중앙값 > 0) 2. OOS MDD ≤ 25% 3. OOS PF ≥ 1.2 4. OOS 거래 ≥ 100건
5. 과반 심볼에서 수익 6. walk-forward 과반 fold 수익 7. 청산 0건
8. 수수료 2배에서도 수익 9. Monte Carlo 자본 -50% 확률 < 5%
10. **유니버스 과반 심볼 OOS 수익** 11. **유니버스 청산 심볼 0개**

- 10개 이상 → `실전 적용 가능 (소액부터)`
- 6~9개 → `추가 연구 필요`
- 5개 이하 → `폐기`

10·11번이 핵심입니다. 바스켓 8종목에서만 되는 전략은 파라미터를 그 8종목에 맞춘
것일 뿐이고, 전 종목에서 재현돼야 진짜 엣지입니다.

## 6. 실거래 봇 연결 시 주의

- `strategy_config.yaml` 의 `entry / grid / risk` 가 그대로 봇 규칙입니다.
- **PCA는 고정된 loading을 영원히 쓰면 안 됩니다.** 라이브에서는 직전 N개월 봉으로
  주기적으로(예: 월 1회) scaler+PCA를 재적합하고, 임계값은 `train_quantiles` 처럼
  **그 시점 분포의 분위수**로 다시 계산하십시오. `pca_model.PCAModel` 을 그대로 재사용하면 됩니다.
- 백테스트는 그리드/TP를 지정가로 가정합니다. 실봇도 지정가로 걸어야 결과가 재현됩니다.
- 물타기 전략은 **최대 명목가 한도**가 생명입니다. `position_pct`, `max_entries` 를 임의로
  키우면 청산 위험이 비선형으로 커집니다.

## 7. 모듈 구조

```
src/universe.py      상장 종목 조회 + 유동성/상장기간 필터
src/scan.py          스트리밍 유니버스 스캔, 통합 통계, 바스켓 선정, 전 종목 재검증
src/indicators.py    RSI/BB/ATR/EMA/ADX (Wilder), 전부 causal
src/features.py      11개 Feature + forward return 라벨 + 시간순 분할
src/pca_model.py     train-only PCA, 부호 고정, 분위수 임계값, loading 해석 리포트
src/analysis.py      구간별 forward-return 통계, 조건별 edge
src/signal.py        진입 조건 조합 + A~F 필터 스택
src/grid.py          ATR 그리드 간격/비중
src/risk.py          사이징, TP/SL, 청산가
src/backtest.py      이벤트 기반 엔진 (수수료/슬리피지/펀딩/청산/거래로그)
src/metrics.py       성과지표 + robust_score
src/optimization.py  탐색 / 민감도 / plateau 선택
src/walk_forward.py  fold 생성, fold별 PCA·파라미터 재적합, OOS 스티칭
src/monte_carlo.py   거래 재표본 + 비용·지연 스트레스
src/regimes.py       시장 국면 라벨링 및 귀속
src/charts.py        그래프 출력
src/run_research.py  STEP 1~16 오케스트레이션
```
