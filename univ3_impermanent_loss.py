from typing import Optional
from dojo.agents import BaseAgent
from dojo.environments.uniswapV3 import UniV3Obs
from dojo.observations.uniswapV3 import UniV3Obs
from Garch_Model import GARCHModel
from decimal import Decimal
from datetime import timedelta
import pandas as pd
import numpy as np
import math
from .Uniswap_Volume import calculate_volume


class ImpermanentLossAgent(BaseAgent):

    def __init__(self, initial_portfolio: dict , counter , garch_model, p , q , regression_model , data , vol_start , vol_end , name: Optional[str] = None):
        super().__init__(name=name, initial_portfolio=initial_portfolio)
        self.hold_portfolio = []
        self.counter = counter
        self.garch_model = garch_model
        self.p = p
        self.q = q
        self.regression_model = regression_model
        self.data = data[["log_returns"]].dropna()        
        self.vol_start = vol_start
        self.vol_end = vol_end
        self.predictions = pd.DataFrame(columns = ['start_time', 'end_time', 'price', 'log_returns' , "predicted_volatility"])
        self.fee = 0.05

    def _pool_wealth(self, obs: UniV3Obs, portfolio: dict) -> float:
        """Calculate the wealth of a portfolio denoted in the y asset of the pool.

        :param portfolio: Portfolio to calculate wealth for.

        :raises ValueError: If agent token is not in pool.
        """
        wealth = 0
        if len(portfolio) == 0:
            return wealth

        pool = obs.pools[0]
        pool_tokens = obs.pool_tokens(pool=pool)
        for token, quantity in portfolio.items():
            if token not in pool_tokens:
                raise ValueError(f"{token} not in pool, so it can't be priced.")
            price = obs.price(token=token, unit=pool_tokens[1], pool=pool)
            wealth += quantity * price
        return wealth

    def reward(self, obs: UniV3Obs) -> float:
        print("COUNTERCOUNTERCOUNTERCOUNTERCOUNTERCOUNTERCOUNTER" , self.counter)
        """Impermanent loss of the agent denoted in the y asset of the pool."""
        token_ids = self.erc721_portfolio().get("UNI-V3-POS", [])
        univ3_obs = UniV3Obs(obs.pools, obs.backend) 
        pool_name = "USDC/WETH-0.05"  # Replace with the actual pool name
        token_price = Decimal(univ3_obs.price(token="WETH", unit="USDC", pool=pool_name))
        self.pool_id = obs.pools[0]

        self.counter += 1  # Increment the counter each time this function is called
        self.vol_end += timedelta(seconds=12)

        #calculate volume
        swap_data = calculate_volume(self.vol_start, self.vol_end, self.pool_id)
        total_vol = swap_data["WETH"] * token_price + swap_data["USDC"]
        
        #calculate % fee with default == 0.05%
        if self.counter % 5 == 0:
            self.fee = self._action_on_calls(token_price) 

        # Calculate total uncollected fees in USDC equivalent
        if not self.hold_portfolio:
            self.hold_portfolio = obs.lp_quantities(token_ids)
        hold_wealth = self._pool_wealth(obs, self.hold_portfolio)
        lp_wealth = self._pool_wealth(obs, obs.lp_portfolio(token_ids))
        if hold_wealth == 0:
            return 0.0
        return lp_wealth - hold_wealth + (Decimal(self.fee) * Decimal(total_vol))
 
    def _action_on_calls(self , token_price):
        print("ACTIONACTIONACTIONACTIONACTIONACTIONACTIONACTIONACTIONACTIONACTIONACTIONACTIONACTIONACTION" , self.counter)
        if self.counter == 5:
            self.predictions.loc[len(self.predictions)] = [self.vol_start, self.vol_end , token_price, np.nan, np.nan]
            return 0.05

        most_recent_price = self.predictions.iloc[-1]['price']
        print(most_recent_price , token_price)
        print(self.predictions)
        recent_log_returns = math.log(most_recent_price / token_price) * 100000
        garch_model_instance = GARCHModel(self.data, self.garch_model, self.regression_model, self.vol_start, self.vol_end, self.p, self.q)
        self.garch_model , fee , vol = garch_model_instance.run()
        self.predictions.loc[len(self.predictions)] = [self.vol_start, self.vol_end , token_price, recent_log_returns , vol]

        #update start_time to be same as end_time
        self.vol_start = self.vol_end
        print(self.predictions)        
        return fee
            
#Pass datafile, garch model, regression model into garch model that will output a new garch model(set garch model to new model) and total volume
#Use fee * total volume minute
#Use that value as total fee and calculate impermanant loss
        