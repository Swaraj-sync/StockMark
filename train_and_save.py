import yfinance as yf
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb


data = yf.download('NVDA', start='2023-01-01', end='2023-12-31')


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
    return data.dropna()


data = add_features(data)

data['Target'] = np.where(data['Close'].shift(-5) > data['Close'], 1, 0)
data = data.iloc[:-5]


feature_cols = ['SMA_50', 'EMA_20', 'RSI', 'MACD', 'Signal', 'OBV', 'Price_Range', 'Volume_Change', 'Lag1_Close', 'Lag2_Close']
X = data[feature_cols]
y = data['Target']

train_size = int(len(X) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


selector = SelectKBest(score_func=f_classif, k='all')
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected  = selector.transform(X_test_scaled)


tscv = TimeSeriesSplit(n_splits=5)
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200],
    'colsample_bytree': [0.8, 0.9, 1.0]
}
xgb_model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss')
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=tscv, scoring='accuracy', verbose=0)
grid_search.fit(X_train_selected, y_train)
best_model = grid_search.best_estimator_


with open("model.pkl", "wb") as f_model:
    pickle.dump(best_model, f_model)
with open("scaler.pkl", "wb") as f_scaler:
    pickle.dump(scaler, f_scaler)
with open("selector.pkl", "wb") as f_selector:
    pickle.dump(selector, f_selector)

print("Artifacts saved: model.pkl, scaler.pkl, selector.pkl")
