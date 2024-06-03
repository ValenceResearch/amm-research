from dateutil import parser as dateparser
from dojo.dataloaders import UniV3Loader
from dojo.dataloaders.formats import UniV3Burn, UniV3Mint, UniV3Swap

def calculate_volume(start_time, end_time, pool_id):
    # Load data using UniV3Loader
    dataloader = UniV3Loader(
        env_name="UniV3Env",
        date_range=(start_time, end_time),
        pools=[pool_id]
    )
    events = dataloader._load_data(subset=["Swap", "Mint", "Burn"])

    total_volume = {"USDC": 0, "WETH": 0}

    for event in events:
        if isinstance(event, UniV3Swap) or isinstance(event, UniV3Mint) or isinstance(event, UniV3Burn):
            for i, token in enumerate(["token0", "token1"]):
                total_volume[token] += abs(event.quantities[i])

    return total_volume