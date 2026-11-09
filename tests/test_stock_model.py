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
