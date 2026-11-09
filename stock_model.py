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
    df = yf.download(ticker, start=start_date, multi_level_index=False)
    if df.empty:
        return {"error": f"No data found for ticker {ticker}"}

    scaler = MinMaxScaler(feature_range=(0, 1))
    close_scaled = scaler.fit_transform(df[["Close"]]).flatten()

    model = _train_model(close_scaled, seq_length, num_epochs, hidden_dim)
    future_preds_scaled = _forecast_future(model, close_scaled[-seq_length:], future_days)
    future_preds = scaler.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1))

    future_dates = pd.bdate_range(df.index[-1] + pd.Timedelta(days=1), periods=future_days)

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
    df = yf.download(ticker, start=start_date, multi_level_index=False)
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
