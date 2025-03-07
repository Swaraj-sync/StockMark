import requests
from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta

app = Flask(__name__)


with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('selector.pkl', 'rb') as f:
    selector = pickle.load(f)


class SentimentCache:
    def __init__(self, api_key):
        self.api_key = api_key
        self.cache = {}

    def get_sentiment(self, ticker, date):
        if date in self.cache:
            return self.cache[date]

        try:
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={self.api_key}"
            response = requests.get(url)
            data = response.json()

            scores = []
            for item in data.get('feed', []):
                item_date = pd.to_datetime(item['time_published']).date()
                if item_date == date.date():
                    for ts in item.get('ticker_sentiment', []):
                        if ts['ticker'] == ticker:
                            scores.append(float(ts['ticker_sentiment_score']))

            sentiment = np.mean(scores) if scores else 0.5
            self.cache[date] = sentiment
            return sentiment
        except Exception as e:
            print(f"Error fetching sentiment: {e}")
            return 0.5

sentiment_cache = SentimentCache(api_key="2VFN6A6ODFKMGMVY")

def prepare_realtime_features(ticker):

    end_date = datetime.today()
    start_date = end_date - timedelta(days=60)
    data = yf.download(ticker, start=start_date, end=end_date)


    data['SMA_50'] = data['Close'].rolling(50).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['Lag1_Close'] = data['Close'].shift(1)
    data['Lag2_Close'] = data['Close'].shift(2)
    data['Price_Range'] = data['High'] - data['Low']
    data['Volume_Change'] = data['Volume'].pct_change()


    latest_date = data.index[-1]
    data['Sentiment'] = sentiment_cache.get_sentiment(ticker, latest_date)


    features = ['SMA_50', 'EMA_20', 'RSI', 'Lag1_Close', 'Lag2_Close',
                'Price_Range', 'Volume_Change', 'Sentiment']
    latest_features = data[features].iloc[-1].values.reshape(1, -1)


    scaled_features = scaler.transform(latest_features)
    selected_features = selector.transform(scaled_features)

    return selected_features

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker'].upper()

    try:
        features = prepare_realtime_features(ticker)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        result = {
            'ticker': ticker,
            'prediction': 'UP' if prediction == 1 else 'DOWN',
            'confidence': f"{probability*100:.1f}%",
            'color': 'green' if prediction == 1 else 'red'
        }
    except Exception as e:
        result = {
            'error': f"Error processing {ticker}: {str(e)}"
        }

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)