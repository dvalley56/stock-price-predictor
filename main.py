import streamlit as st
import numpy as np 
import pandas as pd
import yfinance as yf
import datetime as dt
import tensorflow as tf
import matplotlib.pyplot as plt
import altair as alt
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import plotly.express as px

from sklearn.preprocessing import MinMaxScaler

st.set_page_config(
    page_title='Stock Price Predictor', 
    page_icon=':chart_with_upwards_trend:', 
    layout='wide')

st.title('Stock Price Predictor')

symbol = st.sidebar.text_input('Enter the stock symbol (e.g. RELIANCE.NS)', 'RELIANCE.NS')

start = dt.datetime(2015,1,1)
end = dt.datetime.today()

data = yf.download(symbol, start=start, end=end)

st.write('Historical Stock Price Data')
st.table(data.head())

st.line_chart(data['Close'])

data = data.reset_index()
data = data[['Date', 'Open', 'High', 'Low', 'Close']]

train_set = data.iloc[:, 1:2].values
sc = MinMaxScaler(feature_range=(0, 1))
training_scaled = sc.fit_transform(train_set)

X_train = []
y_train = []

for i in range(60, len(training_scaled)):
    X_train.append(training_scaled[i-60:i, 0])
    y_train.append(training_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

model_lstm = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.LSTM(units=50, return_sequences=True),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.LSTM(units=50, return_sequences=True),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.LSTM(units=50),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(units=1)
])

model_lstm.compile(
    optimizer='adam', 
    loss='mean_squared_error',
    metrics=['mae', 'mse']
)

early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1)

model_lstm.fit(
    X_train, 
    y_train, 
    epochs=50, 
    batch_size=32, 
    callbacks=[early_stop_callback]
)

st.write('Model training is completed.')

test_start = dt.datetime(2021,1,1)
test_end = dt.datetime.today()

test_data = yf.download(symbol, start=test_start, end=test_end)

st.write('Historical Stock Price Data (Testing)')
st.write(test_data.head())

test_set = test_data.iloc[:, 1:2].values

dataset_total = pd.concat((data['Open'], test_data['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(test_data) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []

for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = model_lstm.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

model_prophet = Prophet()
model_prophet.fit(data.rename(columns={'Date': 'ds', 'Close': 'y'}))

future = model_prophet.make_future_dataframe(periods=365)
forecast = model_prophet.predict(future)

st.write('Actual vs Predicted')
st.write('Stock Price Prediction - Prophet Model')
fig = plot_plotly(model_prophet, forecast, xlabel='Date', ylabel='Price')
st.plotly_chart(fig)

st.write('Actual vs Predicted - LSTM Model')

df = pd.DataFrame({'Actual': test_set.flatten(), 'Predicted': predicted_stock_price.flatten()}, index=test_data.index)
# use plotly express
fig = px.line(df, title='Actual vs Predicted - LSTM Model')
st.plotly_chart(fig)

