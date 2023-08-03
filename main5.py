import copy
import math
import warnings

import pandas as pd
import matplotlib.pyplot as plt
from binance.client import Client
import statsmodels.api as sm
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from pmdarima.arima import auto_arima

KEY = ""
SECRET_KEY = ""
SYMBOL = "BTCUSDT"
WARN = ""


def warning_handler(message, category, filename, lineno, file=None, line=None):
    global WARN
    WARN = str(message)


def get_data_symbol(symbol,  key,secret_key):
    client = Client(key,secret_key)
    klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1HOUR, "1 Jan, 2015")
    df_SYMBOL = pd.DataFrame(klines)
    df_SYMBOL.to_csv(symbol + '.csv', index=None, header=True)


def analytics_plots_diff_period(n, df):
    df1diff = df.diff(periods=n).dropna()

    m = int(len(df1diff.index) / 2 + 1)
    r1 = sm.stats.DescrStatsW(df1diff[m:])
    r2 = sm.stats.DescrStatsW(df1diff[:m])
    print('p-value: ', sm.stats.CompareMeans(r1, r2).ttest_ind()[1])

    df1diff.plot(figsize=(12, 6))
    plt.show()

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(df1diff.values.squeeze(), lags=25, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(df1diff, lags=25, ax=ax2)
    plt.show()

def stationary_test(df):
    adft = adfuller(df, autolag="AIC")
    kpsst = kpss(df)
    adf_results = pd.DataFrame(
        {"Values": [adft[0], adft[1], adft[2], adft[3], adft[4]['1%'], adft[4]['5%'], adft[4]['10%']],
         "Metric": ["Test Statistics", "p-value", "No. of lags used", "Number of observations used",
                    "critical value (1%)", "critical value (5%)", "critical value (10%)"]})
    print("-------------------------------")
    print(adf_results)
    kpss_results = pd.DataFrame(
        {"Values": [kpsst[0], kpsst[1], kpsst[2], kpsst[3]['1%'], kpsst[3]['5%'], kpsst[3]['10%']],
         "Metric": ["Test Statistics", "p-value", "No. of lags used",
                    "critical value (1%)", "critical value (5%)", "critical value (10%)"]})
    print("-------------------------------")
    print(kpss_results)
    print("-------------------------------")




def check_arima(df):
    global WARN
    min_val = 1000000
    par = []
    for d in range(0,3):
        for q in range(0,5):
            for p in range(0,5):

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always", category=UserWarning)
                    warnings.showwarning = warning_handler
                    model = sm.tsa.ARIMA(df['Close'],order=(p,d,q),freq='H',enforce_stationarity=False,enforce_invertibility=False)
                    res = model.fit()
                if len(WARN) > 0:
                    WARN = ""
                    continue
                temp = abs(res.aic)
                print(temp, p,d,q)
                if temp<min_val:
                    par.clear()
                    par.append([p,d,q])
                    min_val=temp
                elif temp==min_val:
                    par.append([p, d, q])
    print(par)
    print(min_val)
    return par[0]

def main_analytics_func(df_set, df_pred):


    #stationary_test(df_set)
    #analytics_plots_diff_period(1, df_set)
    #analytics_plots_diff_period(2, df_set)
    #analytics_plots_diff_period(3, df_set)
    #analytics_plots_diff_period(4, df_set)

    df_pred = df_pred.asfreq('H')
    df_set = df_set.asfreq('H')

    p, d, q = 2,2,2#check_arima(df_set)
    model = sm.tsa.ARIMA(df_set['Close'], order=(p, d, q), freq='H', enforce_stationarity=False,
                         enforce_invertibility=False)

    results = model.fit()
    print(results.aic)

    forecast_steps = len(df_pred)  # Number of steps to forecast
    forecast = results.get_forecast(steps=forecast_steps)
    forecast_values = forecast.predicted_mean
    confidence_intervals = forecast.conf_int()

    # Plotting the data and forecast with confidence intervals
    plt.figure(figsize=(10, 6))
    plt.plot(df_set.index, df_set['Close'], label='Observed Data', marker='o', linestyle='-')
    plt.plot(forecast_values.index, forecast_values, label='Forecast', marker='o', linestyle='-')

    # Filling the area between confidence intervals
    plt.fill_between(forecast_values.index, confidence_intervals.iloc[:, 0], confidence_intervals.iloc[:, 1], alpha=0.3)
    plt.plot(df_pred.index, df_pred['Close'], label="Actual Data", marker='o', linestyle='-')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.title('ARIMA Forecast with Confidence Intervals')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return min(forecast_values), max(forecast_values), forecast_values[0]

if __name__ == '__main__':
    df_BTCUSDT= pd.read_csv(SYMBOL+".csv") #pd.DataFrame(klines)
    df_BTCUSDT.columns = ["Open time", "Open", "High", "Low", "Close", "Volume", "Close time", "Quote asset volume"
    , "Number of trades", "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"]




    X = df_BTCUSDT['Close time']
    X = pd.to_datetime(X,unit='ms')
    Y = df_BTCUSDT['Close']

    df = pd.DataFrame(data={"Close time":X,"Close": Y})
    df = df.set_index('Close time')


    df_set = df.iloc[-1500:-700]
    df_pred = df.iloc[-700:]
    """start_price = -1
    x = []
    y = []
    for i in range(len(df_set)):
        if i == 0:
            start_price = df_set['Close'][i]
        price = df_set['Close'][i]
        percent = 100 - (price / start_price )* 100
        y.append(percent)
        x.append(i)
    # plt.plot(x,y)
    plt.hist(np.array(y), bins=10)
    # df_set.plot()

    plt.show()
    #print(len(df))
    df_set.plot()
    plt.show()"""
    mn, mx, start = main_analytics_func(df_set,df_pred)
    print(mn,mx,start)

    """model = sm.tsa.ARIMA(df_set['Close'], order=(p, d, q), freq='H', enforce_stationarity=False,
                         enforce_invertibility=False)

    results = model.fit()
    print(results.aic)

    forecast_steps = len(df_pred)  # Number of steps to forecast
    forecast = results.get_forecast(steps=forecast_steps)
    forecast_values = forecast.predicted_mean
    confidence_intervals = forecast.conf_int(alpha=0.01)

    # Plotting the data and forecast with confidence intervals
    plt.figure(figsize=(10, 6))
    plt.plot(df_set.index, df_set['Close'], label='Observed Data', marker='o', linestyle='-')
    plt.plot(forecast_values.index, forecast_values, label='Forecast', marker='o', linestyle='-')

    # Filling the area between confidence intervals
    plt.fill_between(forecast_values.index, confidence_intervals.iloc[:, 0], confidence_intervals.iloc[:, 1], alpha=0.3)
    plt.plot(df_pred.index, df_pred['Close'], label="Actual Data", marker='o', linestyle='-')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.title('ARIMA Forecast with Confidence Intervals')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()"""
