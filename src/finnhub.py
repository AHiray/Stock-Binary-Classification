import requests
from datetime import timedelta, date, datetime
import datetime
from apiKeys import finnAPI
import pandas as pd

def cleanData(r):
    r = r.json()
    for i in r:
        i['date'] = datetime.datetime.fromtimestamp(i['datetime'])
        i.pop('category')
        i.pop('id')
        i.pop('datetime')
        i.pop('related')
        i.pop('source')
    return r


class FinnhubMarket():
    def marketNews(self):
        r = requests.get('https://finnhub.io/api/v1/news?category=general&token={}'.format(finnAPI))
        r = cleanData(r)
        return r


class Finnhub():
    def __init__(self, stk):
        self.stk = stk

    def companyNews(self):
        today = date.today()
        prevDate = today - timedelta(days=30)
        r = requests.get(
            'https://finnhub.io/api/v1/company-news?symbol={}&from={}&to={}&token={}'.format(self.stk, prevDate, today,
                                                                                             finnAPI))
        r = cleanData(r)
        return r

    def newsSentiment(self):
        r = requests.get(
            'https://finnhub.io/api/v1/news-sentiment?symbol={}&token=c08c1af48v6plm1el0qg'.format(self.stk))
        return r.json()

    def recommendationTrend(self):
        analysis = pd.read_html("https://finance.yahoo.com/quote/{0}/holders?p={1}".format(self.stk, self.stk))
        r = requests.get('https://finnhub.io/api/v1/stock/recommendation?symbol={}&token={}'.format(self.stk, finnAPI))
        r = r.json()
        r.reverse()
        return r, analysis

    def peerCompanies(self):
        r = requests.get('https://finnhub.io/api/v1/stock/peers?symbol={}&token={}'.format(self.stk, finnAPI))
        return r.json()


f = Finnhub("AAPL")
f.recommendationTrend()