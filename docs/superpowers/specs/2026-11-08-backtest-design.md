# Backtest for Stock-Market-Forecaster

## Problem

The repo has two predictive models but no way to evaluate historical accuracy:

- `market_model.py` — a RandomForest classifier predicting next-day S&P 500 direction (up/down). It already contains backtest logic, but it's nested inside `marketPredict()`, duplicated (the predict/backtest functions and model are defined twice), and the precision scores it computes are discarded rather than surfaced.
- `stock_model.py` — an LSTM that forecasts a ticker's close price N days ahead. It has no backtest at all; it only forecasts forward into unknown future dates.

Additionally, the repo is missing a `requirements.txt`, and `yfinance`/`torch` are not installed in the current environment, so nothing in the repo can currently run.

## Goals

- A runnable `backtest.py` script that evaluates both models against historical data and prints accuracy metrics.
- Verify the whole thing actually runs end-to-end in the current environment (install missing deps, fix any bugs encountered).
- Preserve existing behavior of `marketPredict()` and `get_stock_forecast()` used by `app.py` — no change to the app's user-facing behavior.

## Non-goals

- No UI changes to `app.py`.
- No change to model architecture/hyperparameters (LSTM structure, RandomForest params) — backtest evaluates the models as they exist today.
- No CLI argument parsing for the backtest script; scope and depth are fixed constants per the "Standard" depth chosen (see below), not configurable from the command line.

## Design

### 1. `requirements.txt`

Pin `streamlit`, `yfinance`, `torch`, `scikit-learn`, `pandas`, `numpy`, `matplotlib` so the environment is reproducible.

### 2. `market_model.py` refactor

Lift the nested `predict(train, test, predictors, model)` and `backtest(data, model, predictors, start, step)` functions (currently defined twice, inline, inside `marketPredict()`) to module level. Remove the accidental duplicate `model.fit(...)` call. Add threshold support to `predict` (defaults to 0.5 / `model.predict`, else uses `predict_proba` with the given cutoff) so both the baseline and horizon-enhanced pipelines share one implementation instead of two near-identical copies.

Add `backtest_market_model(threshold=0.6, start=3500, step=250)`:
- Runs the baseline pipeline (`Close, Volume, Open, High, Low` predictors, 100-tree forest) and the horizon-enhanced pipeline (rolling ratio/trend features at horizons `[2, 5, 60, 250, 1000]`, 200-tree forest, 0.6 probability threshold).
- Returns a dict with `baseline_precision`, `enhanced_precision`, the raw prediction frames, and the actual positive (price-up) rate in the test period, so the report can show whether the model beats the base rate.

`marketPredict()` keeps its current signature and return value (`"Invest"` / `"Don't Invest"`), now implemented by calling the shared `backtest()` helper — `app.py` is unaffected.

### 3. `stock_model.py` refactor

Split `get_stock_forecast`'s body into two helpers:
- `_train_model(close_scaled, seq_length, num_epochs, hidden_dim)` — sequence building + train/val split + LSTM training loop (unchanged logic), returns the trained model.
- `_forecast_future(model, last_sequence_scaled, future_days)` — the autoregressive forecast loop with the existing 0.7/0.3 blending, returns scaled predictions.

`get_stock_forecast` becomes: download → scale on full series → `_train_model` → `_forecast_future` → inverse-transform → build response dict. Same inputs/outputs as before.

Add `backtest_stock_model(ticker, start_date="2020-01-01", seq_length=30, num_epochs=200, hidden_dim=64, future_days=5, num_windows=5)`:
- Downloads the ticker's full history once.
- Builds `num_windows` walk-forward cutoffs, spaced `future_days` apart, with the last cutoff positioned so its `future_days`-ahead window lands on the most recent actual data (`cutoffs[j] = L - future_days * (num_windows - j)`).
- Skips cutoffs that don't leave enough history (`cutoff <= seq_length + 50`) — if none remain, returns `{"error": ...}`.
- For each cutoff: fits a **fresh MinMaxScaler on only the pre-cutoff data** (no lookahead leakage), trains a fresh model via `_train_model`, forecasts `future_days` via `_forecast_future`, inverse-transforms, and compares against the actual close prices in that window.
- Per window, computes MAE, RMSE, MAPE, and whether the predicted direction (final predicted price vs. last known price) matched the actual direction.
- Returns a dict with per-window results plus averaged MAE/RMSE/MAPE and overall directional accuracy.

### 4. `backtest.py` (new, repo root)

Runnable via `python backtest.py`. Constants: `TICKERS = ["AAPL", "MSFT"]`, `NUM_WINDOWS = 5`, `FUTURE_DAYS = 5` (matches the "Standard" depth: 2 tickers, 5 windows, full 200 epochs — the existing default in `stock_model.py`, left unchanged).

- `run_market_backtest()` — calls `backtest_market_model()`, prints baseline vs. enhanced precision and the base "up" rate.
- `run_stock_backtests()` — loops `TICKERS`, calls `backtest_stock_model()`, prints per-window MAE/RMSE/MAPE/direction and the averages; prints the error message in place of a crash if a ticker lacks enough history.
- `if __name__ == "__main__":` runs both and prints a combined report.

## Error handling

- `backtest_stock_model` returns `{"error": ...}` (same convention as `get_stock_forecast`) rather than raising, for tickers with insufficient history — `backtest.py` prints the message and continues to the next ticker instead of crashing the whole run.
- Network/data failures from `yfinance` (e.g. empty frame) are handled the same way the existing `get_stock_forecast` handles them.

## Verification

After implementation: install dependencies from `requirements.txt`, run `python backtest.py` end-to-end, confirm it completes without exceptions, and sanity-check the output (precision scores in [0,1], MAE small relative to the stock's price, directional accuracy a plausible fraction).
