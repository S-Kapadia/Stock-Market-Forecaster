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
