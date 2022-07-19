# Import Libraries
from datetime import timedelta, date
from math import floor, sqrt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import streamlit as st




class Modelling:

    def __init__(self):
        self.X, self.Y = self.cleanDF()
        self.Y_test = np.array(self.Y.iloc[-floor(len(self.X) * .05):]).reshape(-1, 1)
        self.Y_train = self.Y.iloc[:-floor(len(self.X) * .05)]
        self.X_test = self.X[-floor(len(self.X) * .05):]
        self.X_train = self.X[:-floor(len(self.X) * .05)]
        self.accuracyList = []

    def getAccuracyList(self):
        return self.accuracyList

    def futureDates(self, num):
        dates = []
        startDate = date.today()
        future = (date.today() + timedelta(days=num))
        while startDate <= future:
            dates.append(startDate)
            startDate += timedelta(days=1)
        return dates

    def predict(self, modName, pred, accuracy, interval):
        if pred == 0:
            pred = "RED"
        else:
            pred = 'green'
        st.write("{0} - {1}.".format(modName, pred))
        st.text("{0} days accuracy- {1}".format(len(self.X_test), accuracy))
        st.text("90% confidence interval: {0}".format(interval))

    def confidenceInterval(self, accuracy):
        margin = 1.645 * sqrt((accuracy * (1 - accuracy) / len(self.X_test)))
        upper = accuracy + margin
        lower = accuracy - margin
        if upper >= 1:
            upper = 100
        if lower <= 0:
            lower = 0
        return lower, upper

    def shouldFlip(self, accuracy):
        lower, upper = self.confidenceInterval(accuracy)
        if upper < .5:
            return True

    def cleanDF(self):
        df = pd.read_csv('data.csv', header=0, index_col=0, parse_dates=True)
        df.insert(0, 'Index Numbered', range(0, 0 + len(df)))
        scaler = MinMaxScaler()
        df['Daily Return'] = df["Daily Return"].fillna(0)
        df['ones'] = [floor(i) + 1 for i in df['Daily Return']]
        df['Weekly Vol'] = self.rollingVolatility(df, 5)
        df['2 Vol'] = self.rollingVolatility(df, 2)
        df['Momentum'] = self.momentum(df, 2)
        df['Rolling Return'] = self.rollingReturn(df, 5)
        X = df[['Daily Return', 'Weekly Vol', '2 Vol', 'Momentum', 'Rolling Return']].shift(0).fillna(
            0).iloc[
            ::-1]
        self.today = X.head(1)
        X.drop(X.head(1).index, inplace=True)
        scaler.fit_transform(X)
        Y = df['ones'].shift(-1).iloc[::-1]
        Y.drop(Y.head(1).index, inplace=True)
        return X, Y

    def cleanDFRegression(self):
        df = pd.read_csv('data.csv', header=0, index_col=0, parse_dates=True)
        df.insert(0, 'Index Numbered', range(0, 0 + len(df)))
        df['Index Numbered'].values.reshape(-1, 1)
        return df

    def rollingMean(self, df):
        return df['Daily Return'].mean()

    def movingMA(self, df, length):
        return df['Daily Return'].rolling(length).mean()

    def momentum(self, df, length):
        return df['5. adjusted close'][0] - df['5. adjusted close'][length]

    def rollingReturn(self, df, length):
        return df['Daily Return'].multiply(1).rolling(length).mean()

    def rollingVolatility(self, df, length):
        return df['Daily Return'].rolling(length).std()

    def rollingSharpe(self, df, length):
        return self.rollingReturn(df, length) / self.rollingVolatility(df, length)



    def logisticRegression(self):
        parameters = {
            'solver': ['newton-cg', 'lbfgs', 'liblinear'],
            'C': [100, 10, 1.0, 0.1, 0.01],
        }
        model = LogisticRegression(max_iter=10000)
        grid_search = GridSearchCV(model, parameters, n_jobs=-1).fit(self.X_train, self.Y_train)
        predTest = grid_search.predict(self.X_test)
        predTom = grid_search.predict(self.today)
        accuracy = metrics.accuracy_score(np.array(predTest), self.Y_test)
        if self.shouldFlip(accuracy):
            accuracy = 1 - accuracy
            predTom = not predTom
        self.accuracyList.append(accuracy)
        self.predict("Logistic Regression", predTom, accuracy, self.confidenceInterval(accuracy))

    def sGD(self):
        parameters = {
            'loss': ('log', 'hinge', 'modified_huber'),
            'penalty': ['l1', 'l2'],
            'alpha': [0.01, 0.001, 0.0001],
            'class_weight': [None, 'balanced']
        }
        model = SGDClassifier(max_iter=10000)
        grid_search = GridSearchCV(model, parameters, n_jobs=-1, verbose=1)
        grid_search.fit(self.X_train, self.Y_train)
        predTest = grid_search.predict(self.X_test)
        predTom = grid_search.predict(self.today)
        accuracy = metrics.accuracy_score(np.array(predTest), self.Y_test)
        if self.shouldFlip(accuracy):
            accuracy = 1 - accuracy
            predTom = not predTom
        self.accuracyList.append(accuracy)
        self.predict("Stochastic Gradient Descent", predTom, accuracy, self.confidenceInterval(accuracy))

    def Kneighbors(self):
        parameters = {
            'n_neighbors': [4, 5, 6, 7, 8, 9, 10],
            'weights': ['uniform', 'distance'],
            'leaf_size': [10, 15, 20, 25, 30, 35, 40, 45, 50],
        }
        model = KNeighborsClassifier()
        grid_search = GridSearchCV(model, parameters, n_jobs=-1, verbose=1)
        grid_search.fit(self.X_train, self.Y_train)
        predTest = grid_search.predict(self.X_test)
        predTom = grid_search.predict(self.today)
        accuracy = metrics.accuracy_score(np.array(predTest), self.Y_test)
        if self.shouldFlip(accuracy):
            accuracy = 1 - accuracy
            predTom = not predTom
        self.accuracyList.append(accuracy)
        self.predict("K-Nearest Neighbors Classification", predTom, accuracy, self.confidenceInterval(accuracy))

    def gaussianBayes(self):
        model = GaussianNB().fit(self.X_train, self.Y_train)
        predTest = model.predict(self.X_test)
        predTom = model.predict(self.today)
        accuracy = metrics.accuracy_score(np.array(predTest), self.Y_test)
        if self.shouldFlip(accuracy):
            accuracy = 1 - accuracy
            predTom = not predTom
        self.accuracyList.append(accuracy)
        self.predict("Gaussian Bayes Classification", predTom, accuracy, self.confidenceInterval(accuracy))

    def randomForest(self):
        parameters = {
            'criterion': ['gini', 'entropy'],
            'n_estimators': [50, 100, 150, 200, 250, 300],
        }
        model = RandomForestClassifier()
        grid_search = GridSearchCV(model, parameters, n_jobs=-1, verbose=1)
        grid_search.fit(self.X_train, self.Y_train)
        predTest = grid_search.predict(self.X_test)
        predTom = grid_search.predict(self.today)
        accuracy = metrics.accuracy_score(np.array(predTest), self.Y_test)
        if self.shouldFlip(accuracy):
            accuracy = 1 - accuracy
            predTom = not predTom
        self.accuracyList.append(accuracy)
        self.predict("Random Forest Classification", predTom, accuracy, self.confidenceInterval(accuracy))

    def adaBoost(self):
        parameters = {
            'n_estimators': [10, 50, 100, 120, 140, 160, 180, 200, 250, 500],
        }
        model = AdaBoostClassifier()
        grid_search = GridSearchCV(model, parameters)
        grid_search.fit(self.X_train, self.Y_train)
        predTest = grid_search.predict(self.X_test)
        predTom = grid_search.predict(self.today)
        accuracy = metrics.accuracy_score(np.array(predTest), self.Y_test)
        if self.shouldFlip(accuracy):
            accuracy = 1 - accuracy
            predTom = not predTom
        self.accuracyList.append(accuracy)
        self.predict("AdaBoost Classification", predTom, accuracy, self.confidenceInterval(accuracy))

    def gradientBoost(self):
        parameters = {
            'n_estimators': [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300],
        }
        model = GradientBoostingClassifier()
        grid_search = GridSearchCV(model, parameters)
        grid_search.fit(self.X_train, self.Y_train)
        predTest = grid_search.predict(self.X_test)
        predTom = grid_search.predict(self.today)
        accuracy = metrics.accuracy_score(np.array(predTest), self.Y_test)
        if self.shouldFlip(accuracy):
            accuracy = 1 - accuracy
            predTom = not predTom
        self.accuracyList.append(accuracy)
        self.predict("Gradient Boost Classification", predTom, accuracy, self.confidenceInterval(accuracy))

    def gaussianProcess(self):
        model = GaussianProcessClassifier()
        model.fit(self.X_train, self.Y_train)
        predTest = model.predict(self.X_test)
        predTom = model.predict(self.today)
        accuracy = metrics.accuracy_score(np.array(predTest), self.Y_test)
        if self.shouldFlip(accuracy):
            accuracy = 1 - accuracy
            predTom = not predTom
        self.accuracyList.append(accuracy)
        self.predict("Gradient Process Classification", predTom, accuracy, self.confidenceInterval(accuracy))

    def mLP(self):
        parameters = {
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'hidden_layer_sizes': [i for i in range(10, 500, 100)]
        }
        model = MLPClassifier(max_iter=10000)
        grid_search = GridSearchCV(model, parameters, n_jobs=-1, verbose=1)
        grid_search.fit(self.X_train, self.Y_train)
        predTest = grid_search.predict(self.X_test)
        predTom = grid_search.predict(self.today)
        accuracy = metrics.accuracy_score(np.array(predTest), self.Y_test)
        if self.shouldFlip(accuracy):
            accuracy = 1 - accuracy
            predTom = not predTom
        self.accuracyList.append(accuracy)
        self.predict("Multi-layer Perceptron Classifier", predTom, accuracy, self.confidenceInterval(accuracy))


    def linearRegression(self, days):
        df = self.cleanDFRegression()
        X = df['Index Numbered'].values.reshape(-1, 1)
        Y = np.log2(df['5. adjusted close'])
        model = LinearRegression().fit(X, Y)
        forecast = np.array(range(len(df), len(df) + days + 1)).reshape(-1, 1)
        forPred = model.predict(forecast)
        fig, ax = plt.subplots()
        ax.plot(df.index, df['5. adjusted close'])
        ax.plot(np.array(self.futureDates(days)), np.exp(forPred) / 10)
        return fig
