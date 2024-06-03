from collections import deque
from decimal import Decimal
from typing import List
import pickle
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
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from agents.Uniswap_Volume import calculate_volume


class GARCHModel:
    def __init__(self, data, garch_model , regression_model , start_time , end_time , p , q ):
        self.data = data
        self.garch_model = garch_model
        self.regression_model = regression_model
        self.start_time = start_time
        self.end_time = end_time
        self.best_p = p
        self.best_q = q
        
    def run(self):
        #Fitting Model and Saving it
        volatility = self.forecast()

        #getting volatility prediction of next step
        volatility = np.array(volatility).reshape(1, -1)
        regression_results = self.regression_model.predict(volatility)

        fee = 1/2 * regression_results

        return self.results , fee[0] , volatility
        
    def forecast(self):
        # get log returns of data stored by csv
        train = self.data
        # build initial garch model using trained data
        self.model = arch.arch_model(train, vol='Garch', p=self.best_p, q=self.best_q)
        model_fit = self.model.fit(disp = "off")
        pred = model_fit.forecast(horizon = 1)
        self.results = self.model.fit(disp = "off")
        return np.sqrt(pred.variance.values[-1][0])
    




