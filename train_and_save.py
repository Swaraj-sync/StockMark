import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import requests
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from tqdm import tqdm  # For progress tracking


class SentimentCache:
    def __init__(self, api_key):
        self.api_key = api_key
        self.cache = {}

    def get_sentiment(self, ticker, date):
        """Get cached sentiment or fetch from API"""
        if date in self.cache:
            return self.cache[date]

        try:
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={self.api_key}"
            response = requests.get(url)
            response.raise_for_status()
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


sentiment_cache = SentimentCache(api_key="2VFN6A6ODFKMGMVY")  # <-- Replace


def prepare_features(df, ticker, is_train=True):
    """Calculate features without future leakage"""
    df = df.copy()

    # Technical indicators (safe calculation)
    def _sma(series, window):
        return series.expanding(min_periods=1).mean() if is_train else series.rolling(window).mean()

    df['SMA_50'] = _sma(df['Close'], 50)
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # RSI (safe calculation)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).expanding(min_periods=1).mean() if is_train else delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).expanding(min_periods=1).mean() if is_train else (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Lagged features
    df['Lag1_Close'] = df['Close'].shift(1)
    df['Lag2_Close'] = df['Close'].shift(2)

    # Volume features
    df['Price_Range'] = df['High'] - df['Low']
    df['Volume_Change'] = df['Volume'].pct_change()

    # Add sentiment (with progress bar)
    print(f"Fetching sentiment for {len(df)} days...")
    df['Sentiment'] = [sentiment_cache.get_sentiment(ticker, date) for date in tqdm(df.index)]

    return df.dropna()


ticker = "NVDA"
raw_data = yf.download(ticker, start='2020-01-01', end='2023-12-31')  # 4 years for better splits


train_size = int(len(raw_data) * 0.7)  # 70% training
train_raw = raw_data.iloc[:train_size]
test_raw = raw_data.iloc[train_size:]


train_data = prepare_features(train_raw, ticker, is_train=True)
test_data = prepare_features(test_raw, ticker, is_train=False)


train_data['Target'] = (train_data['Close'].shift(-5) > train_data['Close']).astype(int)
test_data['Target'] = (test_data['Close'].shift(-5) > test_data['Close']).astype(int)
train_data = train_data.iloc[:-5].dropna()
test_data = test_data.iloc[:-5].dropna()


feature_cols = ['SMA_50', 'EMA_20', 'RSI', 'Lag1_Close', 'Lag2_Close',
                'Price_Range', 'Volume_Change', 'Sentiment']
X_train = train_data[feature_cols]
y_train = train_data['Target']
X_test = test_data[feature_cols]
y_test = test_data['Target']


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


selector = SelectKBest(f_classif, k='all')
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)


model = xgb.XGBClassifier(
    objective='binary:logistic',
    learning_rate=0.05,
    max_depth=5,
    n_estimators=300,
    early_stopping_rounds=20,
    eval_metric='logloss'
)


tscv = TimeSeriesSplit(n_splits=5)
model.fit(X_train_selected, y_train,
          eval_set=[(X_test_selected, y_test)],
          verbose=False)


y_pred = model.predict(X_test_selected)
print("\nFinal Test Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("selector.pkl", "wb") as f:
    pickle.dump(selector, f)