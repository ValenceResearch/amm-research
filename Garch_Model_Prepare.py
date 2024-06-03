from collections import deque
from decimal import Decimal
from typing import List
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dojo.actions import BaseAction
from dojo.agents import BaseAgent
from dojo.environments.uniswapV3 import UniV3Obs, UniV3Trade
from dojo.policies import BasePolicy
from scipy.stats import shapiro
import arch
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

class GARCHModel:
    def __init__(self, data_file, loaded_model = None):
        self.data_file = data_file
        self.model = None

    def load_data(self):
        #max_data = 1000
        # Load the CSV file into a DataFrame
        self.data = pd.read_csv(self.data_file)
       # self.data = self.data[:max_data]

    def preprocess_data(self):
        self.data = self.data.dropna()
        # Convert the microsecond timestamps to datetime
        self.data['datetime'] = pd.to_datetime(self.data['local_timestamp'], unit='us')

        # Truncate the datetime to the minute
        self.data['datetime_minute'] = self.data['datetime'].dt.floor('T')

        # Drop duplicates based on the truncated datetime, keeping the first occurrence
        self.data = self.data.drop_duplicates(subset='datetime_minute', keep='first')
        # Extract the log returns data
        self.data["middle_price"] = (self.data["ask_price"] + self.data["bid_price"])/2
        self.data["spread"] = (self.data["ask_price"] - self.data["bid_price"])
        #converting data to log returns and scaling it to prevent convergence issues
        self.data['log_returns'] = np.log(self.data['middle_price'].shift(1) / self.data['middle_price']) * 100000
        self.data.dropna(subset=['log_returns'] , inplace=True)
        self.data = self.data.dropna()
        self.data["rolling_predictions"] = [None] * len(self.data)
        self.data = self.data.reset_index(drop=True)



    def run(self):
        # Main point of entry for workflow
        self.load_data()
        self.preprocess_data()

        #Fitting Model and Saving it
        self.fit_model()

        self.regression()

        return self.results , self.regression_model , self.data , self.best_p , self.best_q
    
        #model visualization
        #self.visualization()

    def fit_model(self):
        #fitting model
        best_aic = 1000000000
        # grid-searching optimal model parameters

        for p in range(1, 5):
            for q in range(1, 5):
                self.model = arch.arch_model(self.data["log_returns"], vol='Garch', p=p, q=q)
                results = self.model.fit(disp='off')
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_results = results
                    self.best_p = p
                    self.best_q = q
        self.results = best_results
        # Fit the model
        #print(self.results.summary())
        #print(best_aic)

        #Rolling Forecast
        test_size = len(self.data["log_returns"])
        for i in range(max(self.best_p, self.best_q) , test_size):
            # get log returns of data stored by csv
            train = self.data["log_returns"][:i]
            # build initial garch model using trained data
            self.model = arch.arch_model(train, vol='Garch', p=self.best_p, q=self.best_q)
            model_fit = self.model.fit(disp = "off")
            pred = model_fit.forecast(horizon = 1)
            self.data.loc[i, "rolling_predictions"] = np.sqrt(pred.variance.values[-1][0])
        self.results = self.model.fit(disp = "off")

    def regression(self):
        X = self.data[['rolling_predictions']].iloc[max(self.best_p, self.best_q):].reset_index(drop=True)
        y = self.data['spread'].iloc[max(self.best_p, self.best_q):].reset_index(drop=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Linear Regression Model (for predicting spread)
        model = LinearRegression()
        self.regression_model = model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Evaluating the model
        mse = mean_squared_error(y_test, predictions)
        #print(f'Mean Squared Error: {mse}')
        #print(f'Coefficients: {model.coef_}')
        #print(f'Intercept: {model.intercept_}')
        #r_squared = model.score(X_test, y_test)
        #print("R^2 Score:", r_squared)
        #print(self.data)

