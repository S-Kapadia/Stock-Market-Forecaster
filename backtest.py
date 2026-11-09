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
