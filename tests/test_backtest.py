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
