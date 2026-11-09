# Stock-Market-Assistant

Stock market predicting assistant that can predict short-term stock prices as well as market health in upcoming days.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
flask run
```

The app starts at `http://127.0.0.1:5000`. Open it in a browser for the UI, or hit the API directly.

## API

### `GET /api/market`

Runs the S&P 500 direction classifier and returns tomorrow's prediction. Trained once per calendar day; subsequent requests the same day reuse the cached result.

```json
{ "prediction": "Invest" }
```

### `GET /api/stock/<ticker>?days=N`

Forecasts `<ticker>`'s closing price `N` business days ahead (`N` must be an integer from 1 to 14; omit `days` to default to 5). Results are cached per `(ticker, days, date)` so retraining only happens once per combination per day.

```json
{
  "ticker": "AAPL",
  "last_date": "2026-11-06",
  "future_forecast": [
    { "date": "2026-11-09", "predicted_price": 268.41 }
  ]
}
```

Invalid `days` or an unknown/empty ticker returns a `400` with `{"error": "..."}`. An unexpected upstream failure (e.g. yfinance unreachable) returns a `502` with the same shape — never an unhandled server error.

## Tests

```bash
pip install pytest
pytest
```

## Backtesting

```bash
python backtest.py
```

Evaluates both models against historical data — see `market_model.py`'s `backtest_market_model()` and `stock_model.py`'s `backtest_stock_model()`.
