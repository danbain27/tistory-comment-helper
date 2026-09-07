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

# Bybit USDT 무기한 15분봉 (+ 펀딩비) 다운로드 -> ./data
python fetch_data.py --symbols BTCUSDT ETHUSDT SOLUSDT XRPUSDT BNBUSDT DOGEUSDT ADAUSDT AVAXUSDT \
                     --interval 15 --days 1095 --funding
```

이미 CSV가 있다면 `data/<SYMBOL>_15m.csv` 로 두면 됩니다.
필요 컬럼: `timestamp(ms), open, high, low, close, volume` (초 단위 timestamp, `datetime`
컬럼도 자동 인식).

## 2. 실행

```bash
python -m src.run_research --symbols BTCUSDT ETHUSDT SOLUSDT XRPUSDT BNBUSDT DOGEUSDT ADAUSDT AVAXUSDT \
    --days 1095 --capital 10000 --leverage 2 \
    --fee-taker 0.00055 --fee-maker 0.0002 --slippage-bp 2 --funding 0.0001

# 빠른 확인용 (탐색 공간 축소)
python -m src.run_research --symbols BTCUSDT ETHUSDT --quick

# 데이터가 없는 환경에서 파이프라인만 점검 (합성 데이터, 전략 근거로 사용 불가)
python -m src.run_research --symbols BTCUSDT ETHUSDT SOLUSDT --days 400 --quick --allow-synthetic
```

무결성 테스트:

```bash
python -m tests.test_engine
```

## 3. 출력물

```
results/
  data_quality.csv          결측/중복/갭/이상치 점검
  pca_components.csv        각 PC의 설명력과 top loading 해석
  pca_loadings_<SYM>.csv    PCA loading 원본
  feature_analysis.csv      Feature 구간별 forward return (1/4/8/16봉)
  entry_conditions.csv      개별 조건의 forward-return edge (t값 포함)
  parameter_test.csv        entry / PCA / grid / TP·SL 탐색 전체 결과
  parameter_sensitivity.csv 파라미터 값별 점수 분포 (과최적화 진단)
  filter_stack.csv          A(RSI) -> F(+ADX) 필터 누적 효과 = PCA 기여도 검증
  backtest_results.csv      최종 전략의 train / validation / test 성과
  walkforward.csv           fold별 val -> test 성과와 선택된 파라미터
  walkforward_summary.csv   심볼별 walk-forward OOS 집계
  monte_carlo.csv           거래 재표본 시뮬레이션 결과
  cost_stress.csv           수수료 2배 / 슬리피지 / 진입지연 스트레스
  regime_analysis.csv       시장 국면별 성과
  trades.csv                OOS 거래 로그 (요구 필드 전부 포함)
  REPORT.md                 사람이 읽는 최종 리포트 + 결론
charts/
  equity.png  drawdown.png  monthly_returns.png  trade_distribution.png
  pca_analysis.png  pca_scatter.png  parameter_sensitivity.png
strategy_config.yaml        실봇에 넣을 최종 규칙 + 성과 스냅샷
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

`run_research.py` STEP 16 이 9개 항목을 자동 채점합니다.

1. OOS 수익(심볼 중앙값 > 0) 2. OOS MDD ≤ 25% 3. OOS PF ≥ 1.2 4. OOS 거래 ≥ 100건
5. 과반 심볼에서 수익 6. walk-forward 과반 fold 수익 7. 청산 0건
8. 수수료 2배에서도 수익 9. Monte Carlo 자본 -50% 확률 < 5%

- 8개 이상 → `실전 적용 가능 (소액부터)`
- 5~7개 → `추가 연구 필요`
- 4개 이하 → `폐기`

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
