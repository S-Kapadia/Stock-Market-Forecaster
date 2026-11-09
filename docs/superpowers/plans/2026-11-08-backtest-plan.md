# Repo Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a runnable `backtest.py` that evaluates both existing models (S&P 500 direction classifier, per-ticker LSTM price forecaster) against historical data, and verify the whole repo actually runs in the current environment.

**Architecture:** Refactor the backtest logic that already exists (duplicated, unexposed) inside `market_model.py` into reusable module-level functions plus a new `backtest_market_model()`; add an equivalent walk-forward backtest to `stock_model.py` (`backtest_stock_model()`) by splitting its monolithic training function into reusable pieces; wire both into a new root-level `backtest.py` script. Existing `app.py`-facing functions (`marketPredict()`, `get_stock_forecast()`) keep their current signatures and behavior.

**Tech Stack:** Python 3.11, pandas, numpy, scikit-learn (RandomForestClassifier), PyTorch (LSTM), yfinance, pytest for tests.

## Global Constraints

- `marketPredict()` and `get_stock_forecast()` must keep their existing signatures and return values — `app.py` must not need any changes.
- No CLI argument parsing in `backtest.py` — tickers/windows/depth are fixed module-level constants (`TICKERS = ["AAPL", "MSFT"]`, `NUM_WINDOWS = 5`, `FUTURE_DAYS = 5`), matching the "Standard" depth decision (2 tickers, 5 walk-forward windows, full 200 epochs).
- No change to model architecture or hyperparameters (LSTM structure, RandomForest params, the 0.6 probability threshold, horizons `[2, 5, 60, 250, 1000]`) — the backtest evaluates the models as they exist today.
- `backtest_stock_model` must fit a fresh `MinMaxScaler` on only the pre-cutoff data per window (no lookahead leakage from future prices into the scaler).
- Errors (empty ticker data, insufficient history) are returned as `{"error": ...}` dicts, same convention as the existing `get_stock_forecast`, never raised as exceptions.

---

### Task 1: Environment and test harness setup

**Files:**
- Create: `requirements.txt`
- Create: `conftest.py`

**Interfaces:**
- Produces: a `conftest.py` at repo root so `tests/test_*.py` files can `import market_model`, `import stock_model`, `import backtest` directly (pytest adds any directory containing a `conftest.py` to `sys.path` during collection).

- [ ] **Step 1: Create `requirements.txt`**

```text
streamlit>=1.49
yfinance>=0.2
torch>=2.0
scikit-learn>=1.7
pandas>=2.0
numpy>=2.0
matplotlib>=3.10
```

- [ ] **Step 2: Create empty `conftest.py` at repo root**

```python
```

(An empty file is sufficient — its presence causes pytest to add the repo root to `sys.path`.)

- [ ] **Step 3: Install dependencies**

Run: `pip install -r requirements.txt && pip install pytest`
Expected: both commands complete without errors.

- [ ] **Step 4: Verify the environment**

Run: `python -c "import yfinance, torch, sklearn, pandas, numpy, streamlit; print('ok')"`
Expected: prints `ok` with no `ModuleNotFoundError`.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt conftest.py
git commit -m "Add requirements.txt and test harness conftest"
```

---

### Task 2: Refactor `market_model.py` and add `backtest_market_model()`

**Files:**
- Modify: `market_model.py` (full rewrite)
- Create: `tests/test_market_model.py`

**Interfaces:**
- Produces: `predict(train, test, predictors, model, threshold=0.5) -> pd.DataFrame` with columns `["Target", "Predictions"]`.
- Produces: `backtest(data, model, predictors, start=3500, step=250, threshold=0.5) -> pd.DataFrame` (same columns, concatenated across windows).
- Produces: `_with_horizon_features(sp500: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]`.
- Produces: `backtest_market_model(threshold=0.6, start=3500, step=250) -> dict` with keys `baseline_precision`, `baseline_predictions`, `enhanced_precision`, `enhanced_predictions`, `target_positive_rate`.
- Produces: `marketPredict() -> str`, unchanged return values (`"Invest"` / `"Don't Invest"`).
- Consumes: nothing from other tasks.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_market_model.py`:

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from market_model import predict, backtest, _with_horizon_features, backtest_market_model


def _make_price_frame(n_rows, seed=0):
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2015-01-01", periods=n_rows, freq="D")
    close = 100 + np.cumsum(rng.normal(0, 1, n_rows))
    df = pd.DataFrame({
        "Close": close,
        "Open": close + rng.normal(0, 0.1, n_rows),
        "High": close + abs(rng.normal(0, 0.5, n_rows)),
        "Low": close - abs(rng.normal(0, 0.5, n_rows)),
        "Volume": rng.randint(1_000_000, 5_000_000, n_rows),
    }, index=dates)
    df["Tomorrow"] = df["Close"].shift(-1)
    df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)
    return df


def test_predict_returns_target_and_predictions_columns():
    data = _make_price_frame(300)
    train, test = data.iloc[:250], data.iloc[250:]
    model = RandomForestClassifier(n_estimators=10, random_state=1)
    result = predict(train, test, ["Close", "Volume", "Open", "High", "Low"], model)
    assert list(result.columns) == ["Target", "Predictions"]
    assert len(result) == len(test)


def test_predict_with_threshold_uses_predict_proba():
    data = _make_price_frame(300)
    train, test = data.iloc[:250], data.iloc[250:]
    model = RandomForestClassifier(n_estimators=10, random_state=1)
    result = predict(train, test, ["Close", "Volume", "Open", "High", "Low"], model, threshold=0.9)
    assert result["Predictions"].isin([0, 1]).all()


def test_backtest_concatenates_all_windows():
    data = _make_price_frame(600)
    model = RandomForestClassifier(n_estimators=10, random_state=1)
    result = backtest(data, model, ["Close", "Volume", "Open", "High", "Low"], start=400, step=100)
    assert len(result) == len(data) - 400


def test_with_horizon_features_adds_expected_columns():
    data = _make_price_frame(1100)
    enhanced, new_predictors = _with_horizon_features(data)
    for horizon in [2, 5, 60, 250, 1000]:
        assert f"Close_Ratio_{horizon}" in new_predictors
        assert f"Trend_{horizon}" in new_predictors
    assert not enhanced[new_predictors].isna().any().any()


def test_backtest_market_model_returns_precisions_in_range(monkeypatch):
    data = _make_price_frame(3000)

    class FakeTicker:
        def __init__(self, symbol):
            pass

        def history(self, period="max"):
            df = data.copy()
            df["Dividends"] = 0.0
            df["Stock Splits"] = 0.0
            return df

    import market_model
    monkeypatch.setattr(market_model.yf, "Ticker", FakeTicker)

    result = backtest_market_model(start=1500, step=300)
    assert 0.0 <= result["baseline_precision"] <= 1.0
    assert 0.0 <= result["enhanced_precision"] <= 1.0
    assert 0.0 <= result["target_positive_rate"] <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_market_model.py -v`
Expected: `ModuleNotFoundError` or `ImportError` for `_with_horizon_features` / `backtest_market_model` (they don't exist yet).

- [ ] **Step 3: Rewrite `market_model.py`**

```python
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

BASELINE_PREDICTORS = ["Close", "Volume", "Open", "High", "Low"]
HORIZONS = [2, 5, 60, 250, 1000]


def _load_sp500():
    sp500 = yf.Ticker("^GSPC").history(period="max")
    sp500 = sp500.drop(columns=["Dividends", "Stock Splits"])
    sp500["Tomorrow"] = sp500["Close"].shift(-1)
    sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
    return sp500.loc["2000-01-01":].copy()


def predict(train, test, predictors, model, threshold=0.5):
    model.fit(train[predictors], train["Target"])
    if threshold == 0.5:
        preds = model.predict(test[predictors])
    else:
        probs = model.predict_proba(test[predictors])[:, 1]
        preds = (probs >= threshold).astype(int)
    preds = pd.Series(preds, index=test.index, name="Predictions")
    return pd.concat([test["Target"], preds], axis=1)


def backtest(data, model, predictors, start=3500, step=250, threshold=0.5):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i + step)].copy()
        all_predictions.append(predict(train, test, predictors, model, threshold))
    return pd.concat(all_predictions)


def _with_horizon_features(sp500):
    sp500 = sp500.copy()
    new_predictors = []
    for horizon in HORIZONS:
        rolling_averages = sp500.rolling(horizon).mean()
        ratio_column = f"Close_Ratio_{horizon}"
        sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]
        trend_column = f"Trend_{horizon}"
        sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]
        new_predictors += [ratio_column, trend_column]
    return sp500.dropna(), new_predictors


def backtest_market_model(threshold=0.6, start=3500, step=250):
    sp500 = _load_sp500()

    baseline_model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
    baseline_predictions = backtest(sp500, baseline_model, BASELINE_PREDICTORS, start=start, step=step)
    baseline_precision = precision_score(baseline_predictions["Target"], baseline_predictions["Predictions"])

    sp500_h, new_predictors = _with_horizon_features(sp500)
    enhanced_model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
    enhanced_predictions = backtest(
        sp500_h, enhanced_model, new_predictors, start=start, step=step, threshold=threshold
    )
    enhanced_precision = precision_score(enhanced_predictions["Target"], enhanced_predictions["Predictions"])

    return {
        "baseline_precision": baseline_precision,
        "baseline_predictions": baseline_predictions,
        "enhanced_precision": enhanced_precision,
        "enhanced_predictions": enhanced_predictions,
        "target_positive_rate": sp500_h["Target"].mean(),
    }


def marketPredict():
    sp500 = _load_sp500()
    sp500_h, new_predictors = _with_horizon_features(sp500)
    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
    predictions = backtest(sp500_h, model, new_predictors, threshold=0.6)
    tomorrow_pred = predictions["Predictions"].iloc[-1]
    return "Invest" if tomorrow_pred == 1 else "Don't Invest"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_market_model.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add market_model.py tests/test_market_model.py
git commit -m "Refactor market_model.py into reusable backtest functions"
```

---

### Task 3: Refactor `stock_model.py` and add `backtest_stock_model()`

**Files:**
- Modify: `stock_model.py` (full rewrite)
- Create: `tests/test_stock_model.py`

**Interfaces:**
- Produces: `_train_model(close_scaled: np.ndarray, seq_length, num_epochs, hidden_dim) -> PredictionModel`.
- Produces: `_forecast_future(model, last_sequence_scaled, future_days) -> list[float]` (scaled predictions).
- Produces: `_walk_forward_cutoffs(total_len, future_days, num_windows, min_train) -> list[int]`.
- Produces: `_window_metrics(preds, actual, last_known) -> dict` with keys `mae`, `rmse`, `mape`, `direction_correct`.
- Produces: `backtest_stock_model(ticker, start_date="2020-01-01", seq_length=30, num_epochs=200, hidden_dim=64, future_days=5, num_windows=5) -> dict` with keys `ticker`, `windows` (list of per-window dicts with `cutoff_date`, `mae`, `rmse`, `mape`, `direction_correct`), `avg_mae`, `avg_rmse`, `avg_mape`, `directional_accuracy`; or `{"error": ...}`.
- Produces: `get_stock_forecast(...)`, unchanged signature/return shape.
- Consumes: nothing from other tasks.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_stock_model.py`:

```python
import numpy as np
import pandas as pd

from stock_model import _walk_forward_cutoffs, _window_metrics, backtest_stock_model


def test_walk_forward_cutoffs_spaced_by_future_days():
    cutoffs = _walk_forward_cutoffs(total_len=200, future_days=5, num_windows=5, min_train=50)
    assert cutoffs == [175, 180, 185, 190, 195]


def test_walk_forward_cutoffs_drops_windows_without_enough_history():
    cutoffs = _walk_forward_cutoffs(total_len=120, future_days=5, num_windows=5, min_train=100)
    assert cutoffs == [105, 110, 115]


def test_window_metrics_perfect_prediction_has_zero_error():
    actual = [10.0, 11.0, 12.0]
    metrics = _window_metrics(preds=actual, actual=actual, last_known=9.0)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["mape"] == 0.0
    assert metrics["direction_correct"] is True


def test_window_metrics_detects_wrong_direction():
    metrics = _window_metrics(preds=[9.0], actual=[11.0], last_known=10.0)
    assert metrics["direction_correct"] is False


def test_backtest_stock_model_runs_end_to_end_on_synthetic_data(monkeypatch):
    rng = np.random.RandomState(0)
    n = 150
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    close = 50 + np.cumsum(rng.normal(0, 0.5, n))
    df = pd.DataFrame({"Close": close}, index=dates)

    import stock_model
    monkeypatch.setattr(stock_model.yf, "download", lambda *args, **kwargs: df)

    result = backtest_stock_model(
        "FAKE", seq_length=10, num_epochs=5, hidden_dim=4, future_days=3, num_windows=2
    )

    assert result["ticker"] == "FAKE"
    assert len(result["windows"]) == 2
    assert result["avg_mae"] >= 0.0
    assert 0.0 <= result["directional_accuracy"] <= 1.0


def test_backtest_stock_model_errors_on_insufficient_history(monkeypatch):
    dates = pd.date_range("2023-01-01", periods=20, freq="D")
    df = pd.DataFrame({"Close": np.linspace(50, 51, 20)}, index=dates)

    import stock_model
    monkeypatch.setattr(stock_model.yf, "download", lambda *args, **kwargs: df)

    result = backtest_stock_model(
        "FAKE", seq_length=10, num_epochs=5, hidden_dim=4, future_days=3, num_windows=5
    )
    assert "error" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_stock_model.py -v`
Expected: `ImportError` — `_walk_forward_cutoffs`, `_window_metrics`, `backtest_stock_model` don't exist yet.

- [ ] **Step 3: Rewrite `stock_model.py`**

```python
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(PredictionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def _train_model(close_scaled, seq_length, num_epochs, hidden_dim):
    data = []
    for i in range(len(close_scaled) - seq_length):
        data.append(close_scaled[i:i + seq_length])
    data = np.array(data)

    X = data[:, :-1]
    y = data[:, -1]

    X = torch.from_numpy(X).unsqueeze(-1).float().to(device)
    y = torch.from_numpy(y).unsqueeze(-1).float().to(device)

    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = PredictionModel(input_dim=1, hidden_dim=hidden_dim, num_layers=2, output_dim=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0 and len(X_test) > 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_test)
                val_loss = criterion(val_pred, y_test).item()
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss:.6f}")

    return model


def _forecast_future(model, last_sequence_scaled, future_days):
    model.eval()
    current_seq = torch.tensor(last_sequence_scaled, dtype=torch.float32).view(1, -1, 1).to(device)

    future_preds = []
    with torch.no_grad():
        for _ in range(future_days):
            next_pred = model(current_seq)
            blended = 0.7 * next_pred.item() + 0.3 * current_seq[0, -1, 0].item()
            future_preds.append(blended)
            current_seq = torch.cat(
                (current_seq[:, 1:, :], torch.tensor([[[blended]]], device=device)), dim=1
            )
    return future_preds


def get_stock_forecast(ticker, start_date="2020-01-01", seq_length=30,
                        num_epochs=200, hidden_dim=64, future_days=5):
    df = yf.download(ticker, start=start_date)
    if df.empty:
        return {"error": f"No data found for ticker {ticker}"}

    scaler = MinMaxScaler(feature_range=(0, 1))
    close_scaled = scaler.fit_transform(df[["Close"]]).flatten()

    model = _train_model(close_scaled, seq_length, num_epochs, hidden_dim)
    future_preds_scaled = _forecast_future(model, close_scaled[-seq_length:], future_days)
    future_preds = scaler.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1))

    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=future_days)

    return {
        "ticker": ticker,
        "last_date": str(df.index[-1].date()),
        "future_forecast": [
            {"date": str(date.date()), "predicted_price": float(price[0])}
            for date, price in zip(future_dates, future_preds)
        ],
    }


def _walk_forward_cutoffs(total_len, future_days, num_windows, min_train):
    cutoffs = [total_len - future_days * (num_windows - j) for j in range(num_windows)]
    return [c for c in cutoffs if c > min_train]


def _window_metrics(preds, actual, last_known):
    preds = np.array(preds, dtype=float)
    actual = np.array(actual, dtype=float)
    mae = float(np.mean(np.abs(preds - actual)))
    rmse = float(np.sqrt(np.mean((preds - actual) ** 2)))
    mape = float(np.mean(np.abs((preds - actual) / actual)) * 100)
    direction_correct = bool((actual[-1] > last_known) == (preds[-1] > last_known))
    return {"mae": mae, "rmse": rmse, "mape": mape, "direction_correct": direction_correct}


def backtest_stock_model(ticker, start_date="2020-01-01", seq_length=30,
                          num_epochs=200, hidden_dim=64, future_days=5, num_windows=5):
    df = yf.download(ticker, start=start_date)
    if df.empty:
        return {"error": f"No data found for ticker {ticker}"}

    close = df["Close"].values.flatten()
    cutoffs = _walk_forward_cutoffs(len(close), future_days, num_windows, min_train=seq_length + 50)
    if not cutoffs:
        return {"error": f"Not enough history for {ticker} to backtest"}

    window_results = []
    for cutoff in cutoffs:
        train_close = close[:cutoff]
        actual = close[cutoff:cutoff + future_days]
        if len(actual) < future_days:
            continue

        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(train_close.reshape(-1, 1)).flatten()

        model = _train_model(train_scaled, seq_length, num_epochs, hidden_dim)
        preds_scaled = _forecast_future(model, train_scaled[-seq_length:], future_days)
        preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

        metrics = _window_metrics(preds, actual, last_known=train_close[-1])
        metrics["cutoff_date"] = str(df.index[cutoff - 1].date())
        window_results.append(metrics)

    if not window_results:
        return {"error": f"Not enough history for {ticker} to backtest"}

    return {
        "ticker": ticker,
        "windows": window_results,
        "avg_mae": float(np.mean([w["mae"] for w in window_results])),
        "avg_rmse": float(np.mean([w["rmse"] for w in window_results])),
        "avg_mape": float(np.mean([w["mape"] for w in window_results])),
        "directional_accuracy": float(np.mean([w["direction_correct"] for w in window_results])),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_stock_model.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add stock_model.py tests/test_stock_model.py
git commit -m "Add walk-forward backtest to stock_model.py"
```

---

### Task 4: Add `backtest.py` runner script

**Files:**
- Create: `backtest.py`
- Create: `tests/test_backtest.py`

**Interfaces:**
- Consumes: `backtest_market_model()` from `market_model.py` (Task 2), `backtest_stock_model(ticker, future_days=, num_windows=)` from `stock_model.py` (Task 3).
- Produces: `run_market_backtest() -> dict` (prints a report, returns the `backtest_market_model()` result), `run_stock_backtests() -> dict` (prints a report, returns `{ticker: result}`), module constants `TICKERS`, `NUM_WINDOWS`, `FUTURE_DAYS`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_backtest.py`:

```python
import backtest


def test_run_market_backtest_prints_precisions(monkeypatch, capsys):
    monkeypatch.setattr(
        backtest,
        "backtest_market_model",
        lambda: {
            "baseline_precision": 0.5123,
            "enhanced_precision": 0.6789,
            "target_positive_rate": 0.54,
            "baseline_predictions": None,
            "enhanced_predictions": None,
        },
    )
    backtest.run_market_backtest()
    out = capsys.readouterr().out
    assert "0.5123" in out
    assert "0.6789" in out


def test_run_stock_backtests_handles_error_result(monkeypatch, capsys):
    def fake_backtest_stock_model(ticker, **kwargs):
        if ticker == "AAPL":
            return {
                "ticker": "AAPL",
                "windows": [
                    {"cutoff_date": "2024-01-01", "mae": 1.0, "rmse": 1.2, "mape": 2.0, "direction_correct": True}
                ],
                "avg_mae": 1.0,
                "avg_rmse": 1.2,
                "avg_mape": 2.0,
                "directional_accuracy": 1.0,
            }
        return {"error": f"Not enough history for {ticker} to backtest"}

    monkeypatch.setattr(backtest, "TICKERS", ["AAPL", "MSFT"])
    monkeypatch.setattr(backtest, "backtest_stock_model", fake_backtest_stock_model)

    backtest.run_stock_backtests()
    out = capsys.readouterr().out
    assert "AAPL" in out
    assert "Not enough history for MSFT" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_backtest.py -v`
Expected: `ModuleNotFoundError: No module named 'backtest'`.

- [ ] **Step 3: Create `backtest.py`**

```python
from market_model import backtest_market_model
from stock_model import backtest_stock_model

TICKERS = ["AAPL", "MSFT"]
NUM_WINDOWS = 5
FUTURE_DAYS = 5


def run_market_backtest():
    print("=" * 60)
    print("S&P 500 Market Direction Backtest")
    print("=" * 60)
    results = backtest_market_model()
    print(f"Baseline model precision:         {results['baseline_precision']:.4f}")
    print(f"Horizon-enhanced model precision: {results['enhanced_precision']:.4f}")
    print(f"Actual 'up' rate in test period:  {results['target_positive_rate']:.4f}")
    return results


def run_stock_backtests():
    print("=" * 60)
    print("Per-Ticker LSTM Price Forecast Backtest")
    print("=" * 60)
    all_results = {}
    for ticker in TICKERS:
        print(f"\n--- {ticker} ---")
        result = backtest_stock_model(ticker, future_days=FUTURE_DAYS, num_windows=NUM_WINDOWS)
        if "error" in result:
            print(f"  {result['error']}")
            continue
        for w in result["windows"]:
            print(
                f"  window ending {w['cutoff_date']}: "
                f"MAE={w['mae']:.2f} RMSE={w['rmse']:.2f} MAPE={w['mape']:.2f}% "
                f"dir_correct={w['direction_correct']}"
            )
        print(
            f"  avg MAE={result['avg_mae']:.2f} avg RMSE={result['avg_rmse']:.2f} "
            f"avg MAPE={result['avg_mape']:.2f}% "
            f"directional_accuracy={result['directional_accuracy']:.2%}"
        )
        all_results[ticker] = result
    return all_results


if __name__ == "__main__":
    run_market_backtest()
    print()
    run_stock_backtests()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_backtest.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backtest.py tests/test_backtest.py
git commit -m "Add backtest.py runner script"
```

---

### Task 5: Full end-to-end verification run

**Files:** none (verification only, no commit).

**Interfaces:**
- Consumes: `backtest.py` from Task 4 (which consumes Tasks 2 and 3).

- [ ] **Step 1: Run the full test suite**

Run: `pytest -v`
Expected: all tests from Tasks 2-4 PASS (13 tests total).

- [ ] **Step 2: Run the real backtest end-to-end**

Run: `python backtest.py` (this hits the network via `yfinance` and trains real LSTMs — 2 tickers x 5 windows x 200 epochs, allow several minutes)
Expected: no traceback; output includes:
  - `Baseline model precision:` and `Horizon-enhanced model precision:` both printed as numbers between 0.0 and 1.0
  - For each of `AAPL` and `MSFT`, either 5 window lines with `MAE=`, `RMSE=`, `MAPE=`, `dir_correct=` followed by an `avg MAE=...directional_accuracy=...` line, or (if yfinance rate-limits/fails) a clear `error` message instead of a crash

- [ ] **Step 3: Sanity-check the numbers**

Confirm: both precision values are in `[0, 1]`; each ticker's `avg MAE` is small relative to that ticker's actual share price (e.g. under ~10% of price for AAPL/MSFT); `directional_accuracy` is between 0.0 and 1.0.

If any of this fails, treat it as a bug to fix (not a reason to loosen the assertions) — re-run `pytest -v` after any fix to confirm nothing regressed, then re-run `python backtest.py`.
