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
import pandas as pd
import numpy as np
import os

class GARCHModel:
    def __init__(self, data_file, loaded_model = None):
        self.data_file = data_file
        self.model = loaded_model
        self.results = None
        self.best_p = None
        self.best_q = None

    def load_data(self):
        max_data = 100
        # Load the CSV file into a DataFrame
        self.data = pd.read_csv(self.data_file)
        self.data = self.data[:max_data]

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
        with open('/Users/tonylee/Documents/vanna/vanna-cexdata/models/garch.pkl' , 'wb') as f:
            pickle.dump(self.results, f)
            print(os.getcwd())

        #Model Visualization
        print("visualization")
        self.visualization()

    def fit_model(self):
        print(self.data["log_returns"])
        if not self.model:
        #fitting model
            best_aic = 1000000000
            # grid-searching optimal model parameters

            for p in range(1, 5):
                for q in range(1, 5):
                    print("here")
                    self.model = arch.arch_model(self.data["log_returns"], vol='Garch', p=p, q=q)
                    results = self.model.fit(disp='off')
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_results = results
                        self.best_p = p
                        self.best_q = q
            self.results = best_results
            # Fit the model
            print(self.results.summary())
            print(best_aic)

        #Rolling Forecast
        rolling_predictions = []
        test_size = len(self.data["log_returns"])
        for i in range(max(self.best_p, self.best_q)+1 , test_size + 1):
            # get log returns of data stored by csv
            train = self.data["log_returns"][:i]
            # build initial garch model using trained data
            self.model = arch.arch_model(train, vol='Garch', p=self.best_p, q=self.best_q)
            model_fit = self.model.fit(disp = "off")
            pred = model_fit.forecast(horizon = 1)
            rolling_predictions.append((i, np.sqrt(pred.variance.values[-1][0])))
            print(i)
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

    def update_and_predict(self, new_mid_price):
        # Updating the latest middle price
        new_log_return = abs(np.log(new_mid_price / self.data['middle_price'].iloc[-1])) * 100000

        # Append the new log return to the DataFrame
        new_row = pd.DataFrame({'middle_price': [new_mid_price], 'log_returns': [new_log_return]})
        self.data = pd.concat([self.data, new_row], ignore_index=True)

        # Store the updated DataFrame to CSV
        self.data.to_csv(self.data_file, index=False)

        # Re-fit the model with updated data if necessary
        if not self.model:
            self.fit_model()
        else:
            # Update existing model with new data point
            self.model = arch.arch_model(self.data['log_returns'], vol='Garch', p=self.best_p, q=self.best_q)
            self.results = self.model.fit(last_obs=len(self.data) - 1, update_freq=5, disp='off')

        # Predicting the next variance
        pred = self.results.forecast(horizon=1)
        predicted_variance = np.sqrt(pred.variance.values[-1][0])

        return predicted_variance



#creating model and testing
def create_new_model(data_file):
    model = GARCHModel(data_file=data_file)
    model.run()
    return model

# loading a pre-trained GARCH model and testing
def load_pretrained_model(data_file, model_file):
    with open(model_file, 'rb') as f:
        loaded_model = pickle.load(f)
    model = GARCHModel(data_file=data_file, loaded_model=loaded_model)
    model.run()

data_file = 'datasets/coinbase_quotes_2023-01-01_BTC-USDT.csv'
testmodel = create_new_model(data_file)
print(testmodel.update_and_predict(30000))
