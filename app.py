import re
from datetime import date

from flask import Flask, jsonify, render_template, request

from market_model import marketPredict
from stock_model import get_stock_forecast

app = Flask(__name__)

TICKER_PATTERN = re.compile(r"^[A-Za-z0-9.\-]{1,10}$")

_market_cache = {"date": None, "result": None}
_stock_cache = {}


def _get_market_prediction():
    today = date.today()
    if _market_cache["date"] != today:
        _market_cache["result"] = marketPredict()
        _market_cache["date"] = today
    return _market_cache["result"]


def _get_cached_stock_forecast(ticker, days):
    key = (ticker.upper(), days, date.today())
    if key not in _stock_cache:
        _stock_cache[key] = get_stock_forecast(ticker, future_days=days)
    return _stock_cache[key]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/market")
def api_market():
    try:
        prediction = _get_market_prediction()
    except Exception as exc:
        return jsonify({"error": f"Market prediction failed: {exc}"}), 502
    return jsonify({"prediction": prediction})


@app.route("/api/stock/<ticker>")
def api_stock(ticker):
    if not TICKER_PATTERN.match(ticker):
        return jsonify({"error": "Invalid ticker symbol"}), 400

    days_param = request.args.get("days", "5")
    try:
        days = int(days_param)
    except ValueError:
        return jsonify({"error": "days must be an integer"}), 400
    if not 1 <= days <= 14:
        return jsonify({"error": "days must be between 1 and 14"}), 400

    try:
        result = _get_cached_stock_forecast(ticker, days)
    except Exception as exc:
        return jsonify({"error": f"Forecast failed: {exc}"}), 502

    if "error" in result:
        return jsonify(result), 400

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
