# Competent-human benchmark findings

**Status: FROZEN MANDATORY BENCHMARK.**

Source run: GitHub Actions `37345774948`, head `e90d6076bf3042e37f0dae5f8e1bdfbacfa5ae49`.

This benchmark is intentionally transparent and unfit. It represents a disciplined manual-research projection using recent team opportunity, recent player role, opponent context, roster membership, recent player efficiency, and simple position/league priors.

No Monte Carlo, sportsbook input, or hyperparameter search is used.

## 2025 benchmark

| Outcome | n | Human-style MAE | Phase1B MAE | Human opportunity MAE | Phase1B opportunity MAE |
|---|---:|---:|---:|---:|---:|
| Passing yards | 492 | 65.928 | **63.969** | 7.778 | **7.474** |
| Receptions | 2232 | 1.767 | **1.727** | 2.331 | **2.210** |
| Receiving yards | 2232 | **23.294** | 23.522 | 2.331 | **2.210** |
| Rushing yards | 1000 | 24.753 | **24.381** | 4.386 | **3.892** |

## 2026 Weeks 1-4 burned diagnostics

- passing yards MAE: **67.019**
- receptions MAE: **1.929**
- receiving yards MAE: **25.522**
- rushing yards MAE: **23.288**

## Interpretation

The baseline is deliberately simple, yet it beat Phase1B on 2025 receiving-yard MAE despite having worse target-opportunity error. That is important: a simple human-style efficiency blend extracted signal that the more engineered candidate did not.

The other three heads beat this benchmark only modestly. Therefore merely beating the old V1 engine is no longer sufficient.

## Locked V2 rule

Every future central-projection candidate must be compared against:

1. V1 / frozen incumbent benchmark where available;
2. strong simple statistical baselines;
3. this competent-human benchmark;
4. the best surviving V2 component candidate.

A sophisticated candidate that cannot materially beat the competent-human benchmark on useful player-level accuracy is rejected.

Component correctness remains mandatory: a final-stat improvement caused by opportunity and efficiency errors cancelling each other does not count as a clean architectural win.
