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
