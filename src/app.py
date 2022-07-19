import streamlit as st
import matplotlib.pyplot as plt
from equity import Equity
from finnhub import Finnhub, FinnhubMarket
from volModel import Modelling
import streamlit.components.v1 as components
from stockStats import Statistics
from monteCarlo import MonteCarlo

st.set_page_config(page_title="Test", page_icon=":shark:", layout='wide', initial_sidebar_state='auto')
markNews = st.button("Show Market News")
individualStockAnalysis = False
futureDays = 0
home = True
company = False
models = False


def titleBack(txt, page):
    st.title("{}".format(txt))
    if st.button('Go Back'):
        page = False


if markNews:
    home = False
    titleBack("Market News", markNews)
    col1, col2, col3 = st.beta_columns(3)
    col1.markdown('[MarketWatch](https://www.marketwatch.com/)')
    col2.markdown('[Bloomberg](https://www.bloomberg.com/)')
    col3.markdown('[CNBC](https://www.cnbc.com)')

    fm = FinnhubMarket()
    js = fm.marketNews()
    components.html(
        '''<!-- Macroaxis Widget Start --><script src="https://www.macroaxis.com/widgets/url.jsp?t=62"></script>
        <div class="macroaxis-copyright"></div><!-- Macroaxis Widget End -->''')
    st.title("Top News Of The Week")
    for i in js:
        st.subheader(i['headline'])
        st.write("Date Published", i['date'])
        url = i['url']
        st.markdown('[Read]({})'.format(url))

if home:
    st.title("Individual Stock Analysis")
    stockInput = st.text_input("Stock Ticker")
    col1, col2 = st.beta_columns(2)
    futureDays = st.number_input("Future Forecasting Days", value=252)
    if col1.button("Update Financial Data"):
        stock = stockInput
        eq = Equity(stock)
        individualStockAnalysis = eq.correctTicker()
    if col2.button("Can't find stock ticker?"):
        col2.markdown(
            '[Find Stock Ticker](https://www.marketwatch.com/tools/quotes/lookup.asp?siteID=mktw&Lookup=&Country=us'
            '&Type=All)')

if individualStockAnalysis:
    company = True
    models = True

if company:
    fh = Finnhub(stockInput)
    st.image("https://charts2.finviz.com/chart.ashx?t={}".format(stockInput.lower()))
    st.markdown('[Find Stock Exposure](https://etfdb.com/stock/{}/)'.format(stockInput))
    with st.beta_expander("Company News"):
        cn = fh.companyNews()
        ns = fh.newsSentiment()
        col1, col2, col3 = st.beta_columns(3)
        col1.subheader("Company News Sentiment: {}".format(ns['companyNewsScore']))
        col2.subheader("Average Bullish Sector Sentiment: {}".format(ns['sectorAverageBullishPercent']))
        col3.subheader("Average News Score: {}".format(ns['sectorAverageNewsScore']))
        st.text(
            "Note: Sentiment Analysis only works for individual stocks. It does not find the sentiment of ETFs or "
            "any other investments.")
        for i in cn:
            st.subheader(i['headline'])
            st.write("Date Published", i['date'])
            url = i['url']
            st.markdown('[Read]({})'.format(url))

    with st.beta_expander("Recommendation Trend"):
        col1, col2 = st.beta_columns(2)
        rt, analysis = fh.recommendationTrend()
        st.title("Funds Invested in {0}".format(stockInput))
        for i in analysis:
            for j in range(4, len(i)):
                st.text(i.iloc[j])
        fig, ax = plt.subplots()
        dates = []
        buy = []
        sell = []
        hold = []
        strongBuy = []
        strongSell = []
        for i in rt:
            dates.append(i["period"])
            buy.append(i["buy"])
            sell.append(i["sell"])
            hold.append(i["hold"])
            strongBuy.append(i["strongBuy"])
            strongSell.append(i["strongSell"])
        plt.plot(dates, sell, label="sell")
        plt.plot(dates, hold, label="hold")
        plt.plot(dates, strongBuy, label="Strong Buy")
        plt.plot(dates, strongSell, label="Strong sell")
        plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='left')
        plt.legend()
        col1.pyplot(fig)
        col2.title("Peer Companies:")
        for i in fh.peerCompanies():
            col2.write("\t{}".format(i))

    with st.beta_expander("Statistics and Analysis"):
        stat = Statistics()
        col1, col2 = st.beta_columns(2)
        col1.title("Rolling Sharpe")
        col1.pyplot(stat.rollingSharpe())
        col1.title("Distribution of Returns")
        col1.pyplot(stat.returnDistribution())
        col2.title("Moving Average")
        col2.pyplot(stat.movingAverage())
        col2.title("Relative Volume")
        col2.plotly_chart(stat.relativeVolume())
        st.title("Additional Statistics")
        st.write("Compounding Annual Growth Rate {0}".format(stat.cagr()))
        st.write("Cumulative Return {0}".format(stat.cumulativeReturn()))
        st.write("Skew {0}".format(stat.skew()))
        st.text("Values between -1 and 1 are considered to be fairly symmetrical.")
        st.write("Kurtosis {0}".format(stat.kurtosis()))
        st.text("Values less than 3 are considered mesokurtic.")
        st.title("Box and Whisker Plot")
        st.plotly_chart(stat.boxWhisker())

if models:
    model = Modelling()
    col1, col2 = st.beta_columns(2)
    with col1.beta_expander("Linear Regression"):
        with st.beta_container():
            with st.spinner("Loading Model"):
                st.pyplot(model.linearRegression(int(futureDays)))

    with col2.beta_expander("Classification Models"):
        with st.beta_container():
            with st.spinner("Fitting Models and Training"):
                st.subheader("Tomorrow's forecast using:")
                model.logisticRegression()
                model.sGD()
                model.Kneighbors()
                model.gaussianBayes()
                model.gaussianProcess()
                model.randomForest()
                model.adaBoost()
                model.gradientBoost()
                model.mLP()
                accuracyList = model.getAccuracyList()
                mc = MonteCarlo(accuracyList)
                mc.simulate()
                mc.combinedProbability()
                st.write("Prediction Accuracy For The Majority Decision- {0}%".format(mc.findMajorProb()))
                st.write("While a majority decision is accurate and sufficient, the more partisan a prediction is, "
                         "the higher the accuracy.")
                st.write("The Combined Machine Learning models are {0}% more accurate than a 50% random draw.".format(
                    (float(mc.findMajorProb()) - 50) / 50 * 100))

