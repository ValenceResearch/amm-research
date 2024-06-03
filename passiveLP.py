
from decimal import Decimal
from typing import List

from dojo.actions.base_action import BaseAction
from dojo.agents import BaseAgent
from dojo.environments.uniswapV3 import UniV3Obs, UniV3Quote, UniV3Trade
from dojo.observations import uniswapV3
from dojo.policies import BasePolicy


class PassiveConcentratedLP(BasePolicy):
    """Provide liquidity passively to a pool in the sepcified price bounds."""

    def __init__(
        self, agent: BaseAgent
    ) -> None:
        """Initialize the policy.

        :param agent: The agent which is using this policy.
        :param lower_price_bound: The lower price bound for the tick range of the LP position to invest in.
            e.g. 0.95 means the lower price bound is 95% of the current spot price.
        :param upper_price_bound: The upper price bound for the tick range of the LP position to invest in.
            e.g. 1.05 means the upper price bound is 105% of the current spot price.
        """
        super().__init__(agent=agent)
        self.has_traded = False
        self.has_invested = False
    def fit(self):
        pass
    

    def initial_quote(self, obs: UniV3Obs) -> List[BaseAction]:
        pool_idx = 0
        pool = obs.pools[pool_idx]
        token0, token1 = obs.pool_tokens(pool)
        spot_price = obs.price(token0, token1, pool)
        wallet_portfolio = self.agent.erc20_portfolio()

        token0, token1 = obs.pool_tokens(obs.pools[pool_idx])
        decimals0 = obs.token_decimals(token0)
        decimals1 = obs.token_decimals(token1)
        lower_tick = Decimal(0.01)
        upper_tick = Decimal(1.99)

        lower_price_range = lower_tick * spot_price
        upper_price_range = upper_tick * spot_price
        tick_spacing = obs.tick_spacing(pool)
    
        lower_tick = uniswapV3.price_to_active_tick(
            lower_price_range, tick_spacing, (decimals0, decimals1)
        )
        upper_tick = uniswapV3.price_to_active_tick(
            upper_price_range, tick_spacing, (decimals0, decimals1)
        )

        provide_action = UniV3Quote(
            agent=self.agent,
            pool=pool,
            quantities=[wallet_portfolio[token0], wallet_portfolio[token1]],
            tick_range=(lower_tick , upper_tick),
        )
        self.has_invested = True
        return [provide_action]

    def predict(self, obs: UniV3Obs) -> List[BaseAction]:
        if not self.has_invested:
            return self.initial_quote(obs)
        return []