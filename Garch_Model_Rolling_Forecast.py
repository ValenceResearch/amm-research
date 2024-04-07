from collections import deque
from decimal import Decimal
from typing import List
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   
from arch import arch_model
from dojo.actions import BaseAction
from dojo.agents import BaseAgent
from dojo.environments.uniswapV3 import UniV3Obs, UniV3Trade
from dojo.policies import BasePolicy
from scipy.stats import shapiro
import arch
import pandas as pd
import numpy as np
import os

class GARCHModel:
    def __init__(self, data_file, loaded_model = None):
        self.data_file = data_file
        self.model = loaded_model
        self.results = None

    def load_data(self):
        # Load the CSV file into a DataFrame
        self.data = pd.read_csv(self.data_file) 

    def preprocess_data(self):
        # Extract the log returns data
        self.data["middle_price"] = (self.data["ask_price"] + self.data["bid_price"])/2
        #converting data to log returns and scaling it to prevent convergence issue
        self.data['log_returns'] = abs(np.log(self.data['middle_price'].shift(1) / self.data['middle_price'])) * 100000
        self.data.dropna(subset=['log_returns'] , inplace=True)

    def run(self):
        # Main point of entry for workflow
        self.load_data()
        self.preprocess_data()
        
        #Fitting Model and Saving it
        self.fit_model()
        with open('/Users/andrew/Documents/Dojo_Test/Models/Garch_Model_BTCC.pkl' , 'wb') as f:
            pickle.dump(self.results, f)
            print(os.getcwd())
    

        #Model Visualization
        self.visualization()

    def fit_model(self):
        if not self.model:
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
            self.results = best_results
            # Fit the model
            print(self.results.summary())
            print(best_aic)

        #Rolling Forecast
        rolling_predictions = []
        test_size = len(self.data["log_returns"])
        for i in range(max(p, q)+1 , test_size + 1):
            train = self.data["log_returns"][:i]
            model_fit = self.model.fit(disp = "off")
            pred = model_fit.forecast(horizon = 1)
            rolling_predictions.append((i, np.sqrt(pred.variance.values[-1][0])))
        self.rolling_predictions = pd.DataFrame(rolling_predictions, columns=['Index', 'Predicted_Variance'])

    def visualization(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.data.index, self.data['log_returns'], label='Actual Volatility', color='blue')
        plt.plot(self.rolling_predictions["Index"], self.rolling_predictions["Predicted_Variance"], label='Predicted Volatility', color='red')

        # Adding labels and title
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.title('Actual vs Predicted Volatility')

        # Adding legend
        plt.legend()

        # Displaying the plot
        plt.grid(True)
        plt.show()



#creating model and testing
def create_new_model(data_file):
    model = GARCHModel(data_file=data_file)
    model.run()

# loading a pre-trained GARCH model and testing
def load_pretrained_model(data_file, model_file):
    with open(model_file, 'rb') as f:
        loaded_model = pickle.load(f)
    model = GARCHModel(data_file=data_file, loaded_model=loaded_model)
    model.run()




