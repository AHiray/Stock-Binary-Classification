import requests
import pandas as pd
import streamlit as st
from apiKeys import alphaAPI, alphaBaseUrl


class Equity:

    def checkTicker(self, df, ticker, lower, upper):
        if lower <= upper and df.iloc[lower] <= ticker <= df.iloc[upper]:
            index = len(df[:lower]) + (len(df.iloc[lower:upper]) // 2)
            if ticker == df.iloc[index]:
                st.success("Updated")
                return True
            if ticker > df.iloc[index]:
                return self.checkTicker(df, ticker, index + 1, upper)
            if ticker < df.iloc[index]:
                return self.checkTicker(df, ticker, lower, index - 1)
        else:
            st.error("Invalid Ticker")
            return False

    def correctTicker(self):
        return self.valid

    def __init__(self, stk):
        self.stk = stk
        self.df = pd.read_csv('stockTicker.csv')
        self.valid = self.checkTicker(self.df['Symbol'], self.stk, 0, len(self.df['Symbol']) - 1)
        self.initTimeSeries()

    def initTimeSeries(self):
        if self.valid:
            params = {'function': 'TIME_SERIES_DAILY_ADJUSTED',
                      'symbol': self.stk,
                      'outputsize': 'full',
                      'apikey': alphaAPI}
            response = requests.get(alphaBaseUrl, params=params)
            data = response.json()
            pd.DataFrame(data["Time Series (Daily)"])
            series = pd.DataFrame(data["Time Series (Daily)"])
            series = series.transpose()
            del series['7. dividend amount']
            del series['8. split coefficient']
            del series['4. close']
            series = series.iloc[::-1]
            series['5. adjusted close'] = pd.to_numeric(series["5. adjusted close"], downcast="float")
            series['Daily Return'] = series['5. adjusted close'].pct_change(1)
            series.to_csv('data.csv')
            return series
