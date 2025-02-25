import os
from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest

app = Flask(__name__)

# Load pre-trained artifacts
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("selector.pkl", "rb") as f:
    selector = pickle.load(f)

# Function to compute technical indicators
def add_features(data):
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['RSI'] = 100 - (100 / (1 + (data['Close'].diff(1).clip(lower=0).rolling(14).mean() /
                                     data['Close'].diff(1).clip(upper=0).abs().rolling(14).mean())))
    macd = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    data['MACD'], data['Signal'] = macd, signal
    data['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
    data['Price_Range'] = data['High'] - data['Low']
    data['Volume_Change'] = data['Volume'].pct_change()
    data['Lag1_Close'] = data['Close'].shift(1)
    data['Lag2_Close'] = data['Close'].shift(2)
    return data.dropna()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        ticker = request.form["ticker"].strip().upper()
        stock_data = yf.download(ticker, period="1y")
        if stock_data.empty:
            prediction = "No data available for the given ticker."
        else:
            features = add_features(stock_data)
            latest_sample = features.iloc[-1:].copy()
            sample_scaled = scaler.transform(latest_sample)
            sample_selected = selector.transform(sample_scaled)
            pred = model.predict(sample_selected)[0]
            prediction = f"Prediction: The stock price of {ticker} is expected to {'rise' if pred == 1 else 'fall'} in 5 days."
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
