import os
import json
import pandas as pd
import ollama

# ──────────────────────────────────────────────────────────────────────────────
# 0. Configuration
# ──────────────────────────────────────────────────────────────────────────────
MODELS = ["mistral", "deepseek-r1", "phi4:14b"]
TICKERS = ["AMZN", "MSFT"]
DATA_DIR = "."              # where your CSVs live
RESULTS_DIR = "results"     # where JSONs go
TRAIN_SPLIT = 0.7           # 70/30 train/test split

os.makedirs(RESULTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Utility: compute numeric metrics
# ──────────────────────────────────────────────────────────────────────────────
def compute_metrics(df):
    df = df.sort_values("Date").reset_index(drop=True)
    split = int(len(df) * TRAIN_SPLIT)

    # Trend slope via OLS on (days since start) vs Adj Close
    df["Days"] = df["Date"].map(pd.Timestamp.toordinal) - df["Date"].iloc[0].toordinal()
    X = df["Days"].values.reshape(-1, 1)
    y = df["Adj Close"].values
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression().fit(X[:split], y[:split])
    slope = lr.coef_[0]

    # 30-day moving average on *entire* series
    ma30 = df["Adj Close"].rolling(window=30, min_periods=1).mean().iloc[-1]

    # Annualized volatility (std of daily returns * sqrt(252))
    daily_ret = df["Adj Close"].pct_change().dropna()
    vol = daily_ret.std() * (252 ** 0.5)

    return {
        "slope_per_day": float(slope),
        "last_30d_ma": float(ma30),
        "annualized_volatility": float(vol)
    }

# ──────────────────────────────────────────────────────────────────────────────
# 2. Main loop
# ──────────────────────────────────────────────────────────────────────────────
for ticker in TICKERS:
    # Load data
    csv_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    df = pd.read_csv(csv_path, parse_dates=["Date"])

    # Compute metrics once
    metrics = compute_metrics(df)

    for model in MODELS:
        # 2a. Raw summary
        raw_prompt = (
            f"Here is the time series for {ticker} (Date, Adjusted Close):\n"
            + df.to_string(index=False)
            + "\n\nProvide a concise summary of the key trends."
        )
        raw_resp = ollama.chat(model=model, messages=[{"role": "user", "content": raw_prompt}])
        summary = raw_resp["message"]["content"]

        # Write raw JSON
        raw_out = {
            "dataset": ticker,
            "model": model,
            "summary": summary
        }
        with open(os.path.join(RESULTS_DIR, f"trend_{ticker}_{model}_raw.json"), "w") as f:
            json.dump(raw_out, f, indent=2)

        # 2b. Numeric interpretation
        num_prompt = (
            f"You are a finance expert. Given these metrics for {ticker}:\n"
            f"- Slope per day: {metrics['slope_per_day']}\n"
            f"- 30-day moving average: {metrics['last_30d_ma']}\n"
            f"- Annualized volatility: {metrics['annualized_volatility']}\n\n"
            "Provide a clear, concise interpretation of what these numbers imply."
        )
        num_resp = ollama.chat(model=model, messages=[{"role": "user", "content": num_prompt}])
        interpretation = num_resp["message"]["content"]

        # Write numeric JSON
        numeric_out = {
            "dataset": ticker,
            "model": model,
            "metrics": metrics,
            "interpretation": interpretation
        }
        with open(os.path.join(RESULTS_DIR, f"trend_{ticker}_{model}_numeric.json"), "w") as f:
            json.dump(numeric_out, f, indent=2)

        # 2c. New: investment recommendation
        rec_prompt = (
            f"You are a portfolio manager. Based on the summary and metrics for {ticker}:\n\n"
            f"Summary:\n{summary}\n\n"
            f"Metrics:\n"
            f"- Slope per day: {metrics['slope_per_day']}\n"
            f"- 30-day MA: {metrics['last_30d_ma']}\n"
            f"- Annualized volatility: {metrics['annualized_volatility']}\n\n"
            "Provide a single, concise, actionable recommendation for a default long-term portfolio "
            "(e.g., buy, hold, sell, or specific adjustment), with a brief rationale."
        )
        rec_resp = ollama.chat(model=model, messages=[{"role": "user", "content": rec_prompt}])
        recommendation = rec_resp["message"]["content"]

        # Write recommendation JSON
        rec_out = {
            "dataset": ticker,
            "model": model,
            "recommendation": recommendation
        }
        safe_name = model.replace(":", "_")   # "phi4:14b" → "phi4_14b"
        fname = f"trend_{ticker}_{safe_name}_recommendation.json"
        with open(os.path.join(RESULTS_DIR, fname), "w") as f:
            json.dump(rec_out, f, indent=2)


        print(f"Finished {ticker} + {model}")

print("All done.")
