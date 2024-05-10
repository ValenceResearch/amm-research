"""Global variables for the dashboard."""
import datetime
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Union

import jsons

from dojo.vis import variables


def random_continue(value):
    """TODO."""
    return value * 0.5 + value * (1.0 - (random.random() - 0.5)) * 0.5


@dataclass
class AgentInfo:
    """TODO."""

    name: str
    address: str


@dataclass
class Bookmark:
    """Users can create bookmarks by clicking in the graph."""

    name: str
    block: int


@dataclass
class Action:
    """Holds information about taken actions."""

    info: str = None

    __annotations__ = {"info": str}


@dataclass
class Liquidities:
    """Holds the wallet as well as the LP portfolio."""

    lp_quantities: Dict[str, float]
    lp_fees: Dict[str, float]
    wallet: Dict[str, float]

    __annotations__ = {
        "lp_quantities": Dict[str, float],
        "lp_fees": Dict[str, float],
        "wallet": Dict[str, float],
    }


@dataclass
class Univ3PoolData:
    """Data class to store UniV3 pool data."""

    price: Union[float, None]
    liquidity: Union[int, None]
    __annotations__ = {"price": Union[float, None], "liquidity": Union[int, None]}


@dataclass
class AaveV3PoolData:
    """Data class to store AaveV3 pool data."""

    token_prices: Union[Dict[str, int], None]
    __annotations__ = {"token_prices": Union[Dict[str, int], None]}


@dataclass
class AaveV3Data:
    """Data class to store AaveV3 user data."""

    totalCollateral: float = None
    totalDebt: float = None
    availableBorrows: float = None
    liquidationThreshold: float = None
    ltv: float = None
    healthFactor: float = None


@dataclass
class AgentData:
    """Data class to store agent data."""

    reward: Union[float, None]
    actions: Union[List[Action], None]
    liquidities: Union[Liquidities, None]
    aaveV3data: Union[AaveV3Data, None]
    __annotations__ = {
        "reward": Union[float, None],
        "actions": Union[List[Action], None],
        "liquidities": Union[Liquidities, None],
        "aaveV3data": Union[AaveV3Data, None],
    }


@dataclass
class BlockData:
    """Per block data."""

    univ3_pooldata: Dict[str, Univ3PoolData]
    aaveV3_pooldata: AaveV3PoolData
    agentdata: List[AgentData]
    signals: Dict[str, float]
    __annotations__ = {
        "univ3_pooldata": Dict[str, Univ3PoolData],
        "aaveV3_pooldata": AaveV3PoolData,
        "agentdata": List[AgentData],
        "signals": Dict[str, float],
    }


@dataclass
class PoolInfo:
    """Data class to store pool info."""

    name: str
    token0: str
    token1: str
    fee: float
    __annotations__ = {"name": str, "token0": str, "token1": str, "fee": float}


@dataclass
class Params:
    """Simulation params."""

    progress_value: Union[float, None]
    agents: Union[List[AgentInfo], None]
    pool_info: List[PoolInfo]
    environments: List[str]
    start_date: Union[datetime.datetime, None]
    end_date: Union[datetime.datetime, None]
    signal_names: Set[str] = field(default_factory=set)

    __annotations__ = {
        "progress_value": Union[float, None],
        "agents": Union[List[AgentInfo], None],
        "pool_info": List[PoolInfo],
        "environments": List[str],
        "start_date": Union[datetime.datetime, str, None],
        "end_date": Union[datetime.datetime, str, None],
        "signal_names": Set[str],
    }


@dataclass
class Data:
    """Full data."""

    blockdata: BlockData
    params: Params
    bookmarks: List[Bookmark]
    __annotations__ = {
        "blockdata": Dict[int, BlockData],
        "params": Params,
        "bookmarks": List[Bookmark],
    }


def _def_value_data():
    """Helper function."""
    univ3_pooldata = {}
    agentdata = AgentData(reward=None, actions=[], liquidities=None)
    return BlockData(univ3_pooldata=univ3_pooldata, agentdata=[agentdata], signals={})


def _def_value_params():
    """Helper function."""
    return Params(
        progress_value=None,
        agents=[],
        pool_info=[],
        environments=[],
        start_date=None,
        end_date=None,
    )


data = Data(
    params=Params(
        progress_value=1,
        agents=[],
        pool_info=[],
        environments=[],
        start_date=None,  # datetime.datetime(1900, 1, 1),
        end_date=None,  # datetime.datetime(1900, 1, 20),
    ),
    blockdata=defaultdict(_def_value_data),
    bookmarks=[],
)

# bookmarks = []


is_demo = False

pid = None


def reset():
    """Set back all variables to initial state."""
    data = Data(
        params=Params(
            progress_value=1,
            agents=[],
            pool_info=[],
            environments=[],
            start_date=None,  # datetime.datetime(1900, 1, 1),
            end_date=None,  # datetime.datetime(1900, 1, 20),
        ),
        blockdata=defaultdict(_def_value_data),
        bookmarks=[],
    )
    variables.data = data


def _from_file(filepath):
    if filepath is not None:
        with open(filepath) as f:
            variables.data = jsons.loads(f.read(), variables.Data)
