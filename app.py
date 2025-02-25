from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# Initialize Flask app
app = Flask(__name__)

# Load pretrained artifacts
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("selector.pkl", "rb") as f:
    selector = pickle.load(f)

# Define technical indicators and feature engineering
def SMA(data, period=20):
    return data['Close'].rolling(window=period).mean()

def EMA(data, period=20):
    return data['Close'].ewm(span=period, adjust=False).mean()

def RSI(data, period=14):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def MACD(data, short_period=12, long_period=26, signal_period=9):
    short_ema = EMA(data, short_period)
    long_ema = EMA(data, long_period)
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal

def OBV(data):
    return (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()

# Function to generate features
def add_features(data):
    data['SMA_50'] = SMA(data, 50)
    data['EMA_20'] = EMA(data, 20)
    data['RSI'] = RSI(data)
    data['MACD'], data['Signal'] = MACD(data)
    data['OBV'] = OBV(data)
    data['Price_Range'] = data['High'] - data['Low']
    data['Volume_Change'] = data['Volume'].pct_change()
    data['Lag1_Close'] = data['Close'].shift(1)
    data['Lag2_Close'] = data['Close'].shift(2)

    # Drop NaN values
    data = data.dropna()

    # Keep only the correct 10 features
    feature_cols = ['SMA_50', 'EMA_20', 'RSI', 'MACD', 'Signal', 'OBV', 'Price_Range', 'Volume_Change', 'Lag1_Close', 'Lag2_Close']
    return data[feature_cols]

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

            # Debugging print statement
            print("Features shape before scaling:", features.shape)

            # Ensure exactly 10 features before scaling
            features = features.iloc[:, :10]  

            sample_scaled = scaler.transform(features.iloc[-1:].copy())
            sample_selected = selector.transform(sample_scaled)
            pred = model.predict(sample_selected)[0]

            prediction = f"Prediction: The stock price of {ticker} is expected to {'rise' if pred == 1 else 'fall'} in 5 days."

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
