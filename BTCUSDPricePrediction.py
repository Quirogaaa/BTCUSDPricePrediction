import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

df = pd.read_csv('BTC-USD.csv')

df = df[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]

df_prophet = df[["Date", "Close"]].copy()
df_prophet.columns = ["ds", "y"]

print(df_prophet)

prophet = Prophet()
prophet.fit(df_prophet)

future = prophet.make_future_dataframe(periods=365)
print(future)

forecast = prophet.predict(future)
forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(200)  

data = forecast.copy()
data = pd.DataFrame(data)
data.tail(200)

prophet.plot(forecast)
plt.gcf().set_size_inches(20, 10)
plt.show()
