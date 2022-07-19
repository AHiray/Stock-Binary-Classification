import matplotlib.pyplot as plt
from volModel import Modelling
from dateutil import relativedelta
import math
from scipy.stats import skew, kurtosis
import plotly.graph_objects as go
import plotly.express as px


class Statistics(Modelling):

    def __init__(self):
        self.df = super().cleanDFRegression().fillna(0)

    #use rolling sharpe method from volModel
    def rollingSharpe(self):
        fig, ax = plt.subplots()
        ax.set_title("Rolling Sharpe (Yearly)")
        ax.plot(super().rollingSharpe(self.df, 252))
        plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='left')
        return fig

    # use ma method from volmodel
    def movingAverage(self):
        fig, ax = plt.subplots()
        ax.set_title("Mean Daily Return Moving Average (Yearly)")
        ax.plot(super().movingMA(self.df, 252))
        plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='left')
        return fig

    def cagr(self):
        endDate = self.df.index[-1]
        startDate = self.df.index[0]
        difference = relativedelta.relativedelta(endDate, startDate).years
        l = self.df['5. adjusted close'].iloc[-1]
        f = self.df['5. adjusted close'].iloc[0]
        cagr = (math.pow(l/f, 1/difference) - 1)
        return "{:.2%}".format(cagr)

    def cumulativeReturn(self):
        return "{:.2%}".format(((self.df['5. adjusted close'].iloc[-1] - self.df['5. adjusted close'].iloc[0]) / self.df['5. adjusted close'].iloc[0]))

    def returnDistribution(self):
        fig, ax = plt.subplots()
        ax.set_title("Distribution of Returns")
        ax.hist(self.df['Daily Return'], bins=int(len(self.df)/20))
        return fig

    def skew(self):
        return "{:.5}".format(skew(self.df['Daily Return']))

    def kurtosis(self):
        return "{:.5}".format(kurtosis(self.df['Daily Return']))

    def boxWhisker(self):
        fig = go.Figure()
        fig.add_trace(go.Box(x=self.df['Daily Return']))
        fig.update_layout(title_text="Box Plot Daily Return")
        return fig

    def relativeVolume(self):
        fig = px.line((self.df['6. volume']/self.df['6. volume'].rolling(252).mean()).dropna(0), title='Relative Volume')
        return fig
