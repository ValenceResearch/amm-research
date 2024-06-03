import logging
from decimal import Decimal
import multiprocessing

logging.basicConfig(format="%(asctime)s - %(message)s", level=20)
from dojo.vis.dashboard import run_app
from agents.uniV3_pool_wealth import UniV3PoolWealthAgent
from dateutil import parser as dateparser
from policies.moving_average import MovingAveragePolicy
from policies.passiveLP import PassiveConcentratedLP
from agents.univ3_impermanent_loss import ImpermanentLossAgent
from dojo.environments import UniV3Env
from dojo.runners import backtest_run

from Garch_Model_Prepare import GARCHModel

def run_dashboard(port):
    run_app(port, mode="prod")

#creating model and testing
def create_new_model(data_file):
    model = GARCHModel(data_file=data_file)
    return model.run()

def main():
    # SNIPPET 1 START
    pools = ["USDC/WETH-0.05"]
    start_time = dateparser.parse("2021-06-21 00:00:00 UTC")
    end_time = dateparser.parse("2021-06-21 12:00:00 UTC")

    data_file = '/Users/andrew/documents/datasets/data/coinbase_quotes_2023-01-01_ETH-USDT.csv'
    garch_model , regression_model , data , p , q = create_new_model(data_file)
    
    vol_start = dateparser.parse("2021-06-21 00:00:00 UTC")
    vol_end = dateparser.parse("2021-06-21 00:00:00 UTC")

    agent2 = ImpermanentLossAgent(
        initial_portfolio={"USDC": Decimal(4000), "WETH": Decimal(100)},
        counter=0,
        garch_model=garch_model,
        p = p,
        q = q,
        regression_model=regression_model,
        data=data,
        vol_start=vol_start,
        vol_end=vol_end,
        name="LPAgent",
    )

    # Simulation environment (Uniswap V3)
    env = UniV3Env(
        date_range=(start_time, end_time),
        agents=[agent2],
        pools=pools,
        backend_type="local",
        market_impact="replay",
    )


    # Policies
    passive_lp_policy = PassiveConcentratedLP(
        agent=agent2
    )
    sim_blocks, sim_rewards = backtest_run(
        env, [passive_lp_policy], dashboard_port=8057, auto_close=True
    )
    # SNIPPET 1 END
    with open("simulation_results.txt", "w") as f:
        f.write(f"Simulation Blocks: {sim_blocks}\n")
        f.write(f"Simulation Rewards: {sim_rewards}\n")


if __name__ == "__main__":
    # Start the dashboard in a separate process
    dashboard_process = multiprocessing.Process(target=run_dashboard, args=(8057,))
    dashboard_process.start()
    # Run the simulation
    main()

    # Wait for the dashboard process to finish
    dashboard_process.join()
    
