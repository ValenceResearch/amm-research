"""Plotly graphs for the dashboard."""
import copy
from enum import Enum

import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots

from dojo.vis import variables

pio.templates["myname"] = go.layout.Template(
    layout=go.Layout(colorway=["#6BD6E3", "#BE97D2", "#FF929C"])
)
pio.templates.default = "myname"


ActionType = Enum("ActionType", ["Trade", "Quote"])

COLORS = ["#BE97D2", "#6BD6E3", "#FEE26C", "#FF929C"]
GLOW_WIDTH = 5
GLOW_COLOR = "rgba(255,255,255,0.1)"


def _add_bookmarks_to_fig(fig, textposition="end"):
    for i, bm in enumerate(variables.data.bookmarks):
        fig.add_vline(
            x=bm.block,
            line_width=1,
            line_dash="dash",
            line_color="#FF67C3",
            label=dict(
                text=f"{i}",
                textposition=textposition,
                textangle=0,
                font=dict(color="#FF67C3"),
            ),
        )
    return fig


def _empty_graph():
    fig = go.Figure()
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x",
    )
    fig.update_layout(font_family="Quicksand")
    fig.update_layout(xaxis=dict(tickformat="d"))
    return fig


def custom_figure(v1, v2, separate_axes=True):
    """TODO."""
    x = copy.copy(sorted(list(variables.data.blockdata.keys())))
    fig = make_subplots(specs=[[{"secondary_y": separate_axes}]])

    def _v_to_traces(x, v1, index=0):
        quantity = v1.lower().rsplit("-")[-1]
        if v1.lower().startswith("agent"):
            agent_name = v1.split("-")[1]
            iagent = [agent.name for agent in variables.data.params.agents].index(
                agent_name
            )
            y = [variables.data.blockdata[i].agentdata[int(iagent)].reward for i in x]
        elif v1.lower().startswith("pool"):
            poolname = v1.split("-", maxsplit=1)[1].rsplit("-", maxsplit=1)[0]
            if quantity == "price":
                y = [
                    variables.data.blockdata[i].univ3_pooldata[poolname].price
                    for i in x
                ]
            elif quantity == "liquidity":
                y = [
                    variables.data.blockdata[i].univ3_pooldata[poolname].liquidity
                    for i in x
                ]
            else:
                raise ValueError(quantity)
        elif v1.lower().startswith("signal"):
            quantity = v1.rsplit("-")[-1]
            y = [variables.data.blockdata[i].signals[quantity] for i in x]
        else:
            x = None
            y = None

        if y is None:
            text = None
        else:
            text = [
                f"<b>{v1}</b><br>block:{xi}<br>value:  {yi}" for xi, yi in zip(x, y)
            ]
        trace1 = go.Scatter(
            x=x,
            y=y,
            line=dict(color=COLORS[index], width=1),
            name=v1,
            showlegend=not separate_axes,
            hoverinfo="x+text",
            hovertext=text,
        )
        trace1glow = go.Scatter(
            x=x,
            y=y,
            line=dict(color=GLOW_COLOR, width=GLOW_WIDTH),
            showlegend=False,
            hoverinfo="skip",
        )
        return trace1, trace1glow

    trace1, trace1glow = _v_to_traces(x, v1, 0)
    trace2, trace2glow = _v_to_traces(x, v2, 1)

    fig.add_trace(trace1, secondary_y=False)
    fig.add_trace(trace1glow, secondary_y=False)
    fig.add_trace(trace2, secondary_y=separate_axes)
    fig.add_trace(trace2glow, secondary_y=separate_axes)
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x",
    )
    fig.update_layout(font_family="Quicksand")
    if separate_axes:
        fig.update_yaxes(title=v1, secondary_y=False, title_font_color=COLORS[0])
        fig.update_yaxes(title=v2, secondary_y=True, title_font_color=COLORS[1])
    if not separate_axes:
        fig.update_layout(
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0
            )
        )
    fig.update_xaxes(title="Block")
    fig.update_layout(xaxis=dict(tickformat="d"))

    fig = _add_bookmarks_to_fig(fig)
    return fig


def positions_graph(selected_pool: str):
    """Plots the current positions of the agent."""
    if len(variables.data.params.pool_info) == 0:
        return _empty_graph()
    keys = copy.copy(list(variables.data.blockdata.keys()))

    # Create figure with secondary y-axis
    subplot_titles = []
    for agent in variables.data.params.agents:
        subplot_titles += [
            f"Tokens in Wallet - {agent.name}",
            f"Tokens in LP Positions - {agent.name}",
            f"Uncollected Fees in LP Positions - {agent.name}",
        ]

    # if not variables.data.params.agents or variables.data.params.agents == []:
    #     variables.data.params.num_agents = 2

    num_rows = len(variables.data.params.agents)

    fig = make_subplots(
        rows=num_rows,
        cols=3,
        subplot_titles=subplot_titles,
        shared_xaxes=True,
        vertical_spacing=0.1,
        specs=[[{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}]]
        * num_rows,
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=400 * len(variables.data.params.agents),
        hovermode="x",
    )
    fig.update_layout(font_family="Quicksand")

    pool_info = [
        pi for pi in variables.data.params.pool_info if pi.name == selected_pool
    ][0]

    token0 = pool_info.token0  # TODO this is wrong and needs to be fixed
    token1 = pool_info.token1  # TODO this is wrong and needs to be fixed

    for iagent, _ in enumerate(variables.data.params.agents):
        blocks = [
            i
            for i in keys
            if variables.data.blockdata[i].agentdata[iagent].liquidities is not None
        ]
        trace_lp_token0 = [
            variables.data.blockdata[block]
            .agentdata[iagent]
            .liquidities.lp_quantities[token0]
            for block in blocks
        ]
        trace_lp_token1 = [
            variables.data.blockdata[block]
            .agentdata[iagent]
            .liquidities.lp_quantities[token1]
            for block in blocks
        ]
        trace_wallet_token0 = [
            variables.data.blockdata[block].agentdata[iagent].liquidities.wallet[token0]
            for block in blocks
        ]
        trace_wallet_token1 = [
            variables.data.blockdata[block].agentdata[iagent].liquidities.wallet[token1]
            for block in blocks
        ]
        trace_fees_token0 = [
            variables.data.blockdata[block]
            .agentdata[iagent]
            .liquidities.lp_fees[token0]
            for block in blocks
        ]
        trace_fees_token1 = [
            variables.data.blockdata[block]
            .agentdata[iagent]
            .liquidities.lp_fees[token1]
            for block in blocks
        ]
        wallet_token0 = go.Scatter(
            x=blocks,
            y=trace_wallet_token0,
            hoverinfo="x+y",
            mode="lines",
            name="wallet token 0",
            line=dict(color=COLORS[0], width=1),
            showlegend=False,
        )
        glow_wallet_token0 = go.Scatter(
            x=blocks,
            y=trace_wallet_token0,
            hoverinfo="skip",
            mode="lines",
            line=dict(color=GLOW_COLOR, width=GLOW_WIDTH),
            showlegend=False,
        )
        lp_token0 = go.Scatter(
            x=blocks,
            y=trace_lp_token0,
            hoverinfo="x+y",
            mode="lines",
            name="lp token 0",
            line=dict(color=COLORS[0], width=1),
            showlegend=False,
        )
        glow_lp_token0 = go.Scatter(
            x=blocks,
            y=trace_lp_token0,
            hoverinfo="skip",
            mode="lines",
            line=dict(color=GLOW_COLOR, width=GLOW_WIDTH),
            showlegend=False,
        )
        wallet_token1 = go.Scatter(
            x=blocks,
            y=trace_wallet_token1,
            hoverinfo="x+y",
            mode="lines",
            name="wallet token 1",
            line=dict(color=COLORS[1], width=1),
            showlegend=False,
        )
        glow_wallet_token1 = go.Scatter(
            x=blocks,
            y=trace_wallet_token1,
            hoverinfo="skip",
            mode="lines",
            line=dict(color=GLOW_COLOR, width=GLOW_WIDTH),
            showlegend=False,
        )
        lp_token1 = go.Scatter(
            x=blocks,
            y=trace_lp_token1,
            hoverinfo="x+y",
            mode="lines",
            name="lp token 1",
            line=dict(color=COLORS[1], width=1),
            showlegend=False,
        )
        glow_lp_token1 = go.Scatter(
            x=blocks,
            y=trace_lp_token1,
            hoverinfo="skip",
            mode="lines",
            line=dict(color=GLOW_COLOR, width=GLOW_WIDTH),
            showlegend=False,
        )
        fees_token0 = go.Scatter(
            x=blocks,
            y=trace_fees_token0,
            hoverinfo="x+y",
            mode="lines",
            name="fees token 0",
            line=dict(color=COLORS[0], width=1),
            showlegend=False,
        )
        glow_fees_token0 = go.Scatter(
            x=blocks,
            y=trace_fees_token0,
            hoverinfo="skip",
            mode="lines",
            line=dict(color=GLOW_COLOR, width=GLOW_WIDTH),
            showlegend=False,
        )
        fees_token1 = go.Scatter(
            x=blocks,
            y=trace_fees_token1,
            hoverinfo="x+y",
            mode="lines",
            name="fees token 1",
            line=dict(color=COLORS[1], width=1),
            showlegend=False,
        )
        glow_fees_token1 = go.Scatter(
            x=blocks,
            y=trace_fees_token1,
            hoverinfo="skip",
            mode="lines",
            line=dict(color=GLOW_COLOR, width=GLOW_WIDTH),
            showlegend=False,
        )

        fig.add_trace(wallet_token0, row=iagent + 1, col=1)
        fig.add_trace(glow_wallet_token0, row=iagent + 1, col=1)

        fig.add_trace(wallet_token1, row=iagent + 1, col=1, secondary_y=True)
        fig.add_trace(glow_wallet_token1, row=iagent + 1, col=1, secondary_y=True)

        fig.add_trace(lp_token0, row=iagent + 1, col=2)
        fig.add_trace(glow_lp_token0, row=iagent + 1, col=2)

        fig.add_trace(lp_token1, row=iagent + 1, col=2, secondary_y=True)
        fig.add_trace(glow_lp_token1, row=iagent + 1, col=2, secondary_y=True)

        fig.add_trace(fees_token0, row=iagent + 1, col=3)
        fig.add_trace(glow_fees_token0, row=iagent + 1, col=3)

        fig.add_trace(fees_token1, row=iagent + 1, col=3, secondary_y=True)
        fig.add_trace(glow_fees_token1, row=iagent + 1, col=3, secondary_y=True)

        fig.update_yaxes(
            row=iagent + 1,
            col=1,
            secondary_y=False,
            title=f"{variables.data.params.pool_info[0].token0}",
            title_font_color=COLORS[0],
        )
        fig.update_yaxes(
            row=iagent + 1,
            col=1,
            secondary_y=True,
            title=f"{variables.data.params.pool_info[0].token1}",
            title_font_color=COLORS[1],
        )
        fig.update_yaxes(
            row=iagent + 1,
            col=2,
            secondary_y=False,
            title=f"{variables.data.params.pool_info[0].token0}",
            title_font_color=COLORS[0],
        )
        fig.update_yaxes(
            row=iagent + 1,
            col=2,
            secondary_y=True,
            title=f"{variables.data.params.pool_info[0].token1}",
            title_font_color=COLORS[1],
        )
        fig.update_yaxes(
            row=iagent + 1,
            col=3,
            secondary_y=False,
            title=f"{variables.data.params.pool_info[0].token0}",
            title_font_color=COLORS[0],
        )
        fig.update_yaxes(
            row=iagent + 1,
            col=3,
            secondary_y=True,
            title=f"{variables.data.params.pool_info[0].token1}",
            title_font_color=COLORS[1],
        )

        fig.update_yaxes(row=iagent + 1, col=1, spikemode="across")
        fig.update_yaxes(row=iagent + 1, col=2, spikemode="across")
        fig.update_yaxes(row=iagent + 1, col=3, spikemode="across")

        fig.update_xaxes(
            row=len(variables.data.params.agents), col=1, title="Block", tickformat="d"
        )
        fig.update_xaxes(
            row=len(variables.data.params.agents), col=2, title="Block", tickformat="d"
        )
        fig.update_xaxes(
            row=len(variables.data.params.agents), col=3, title="Block", tickformat="d"
        )
        fig.update_xaxes(spikemode="across")

        fig = _add_bookmarks_to_fig(fig, textposition="start")

    return fig


def reward_graph(agent_name):
    """Plot reward graph."""

    def actions2string(block, actions):
        result = f"BLOCK: {block}<br>"
        result += "ACTIONS:<br>"
        for action in actions:
            result += f" {action.info}"
        return result

    blocks = sorted(list(variables.data.blockdata.keys()))

    fig = make_subplots(
        rows=1,
        cols=1,
        # subplot_titles=[f"agent {i+1}" for i in range(1)],
        shared_xaxes=True,
        vertical_spacing=0.1,
        specs=[[{"secondary_y": True}]] * 1,
    )
    if agent_name is not None:
        iagent = [i for i, a in enumerate(variables.data.params.agents)][0]

        y0 = [
            variables.data.blockdata[block].agentdata[iagent].reward for block in blocks
        ]
        trace = go.Scatter(
            x=blocks,
            y=y0,
            hoverinfo="skip",
            showlegend=False,
            mode="lines",
            line=dict(color=GLOW_COLOR, width=GLOW_WIDTH),
        )
        fig.add_trace(trace)

        trace = go.Scatter(
            x=blocks,
            y=y0,
            name="reward",
            hoverinfo="x+text",
            showlegend=False,
            hovertext=[
                f"<b>Reward</b><br>{round(variables.data.blockdata[block].agentdata[iagent].reward, 3)}"
                for block in blocks
            ],
            mode="lines",
            line=dict(color=COLORS[0], width=1),
        )
        fig.add_trace(trace)
        blocks_with_actions = [
            block
            for block in blocks
            if len(variables.data.blockdata[block].agentdata[iagent].actions) > 0
        ]
        trace_actions = go.Scatter(
            x=blocks_with_actions,
            y=[
                variables.data.blockdata[block].agentdata[iagent].reward
                for block in blocks_with_actions
            ],
            mode="markers",
            hoverinfo="text",
            marker=dict(
                size=8,
                color=COLORS[0],
                line=dict(width=2, color="rgba(255,255,255,0.2)"),
            ),
            name="actions",
            hovertext=[
                actions2string(
                    block, variables.data.blockdata[block].agentdata[iagent].actions
                )
                for block in blocks_with_actions
            ],
            showlegend=False,
        )
        fig.add_trace(trace_actions)

    names = set()
    fig.for_each_trace(
        lambda trace: trace.update(showlegend=False)
        if (trace.name in names)
        else names.add(trace.name)
    )
    fig.update_layout(
        yaxis=dict(title="Reward"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(title_text="Block")
    fig.update_layout(hovermode="x")
    fig.update_layout(font_family="Quicksand")
    fig = _add_bookmarks_to_fig(fig)
    fig.update_layout(xaxis=dict(tickformat="d"))
    return fig


def price_graph(pool: str):
    """Plot price graph."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if pool is not None:
        blocks = sorted(list(variables.data.blockdata.keys()))
        prices = [
            variables.data.blockdata[block].univ3_pooldata[pool].price
            for block in blocks
        ]
        liquidities = [
            variables.data.blockdata[block].univ3_pooldata[pool].liquidity
            for block in blocks
        ]

        trace1 = go.Scatter(
            x=blocks,
            y=prices,
            name="price",
            hoverinfo="x+text",
            hovertext=[f"<b>Price</b><br>{price}" for price in prices],
            mode="lines",
            line=dict(color=COLORS[0], width=1),
        )
        glow_trace1 = go.Scatter(
            x=blocks,
            y=prices,
            hoverinfo="skip",
            mode="lines",
            line=dict(color=GLOW_COLOR, width=GLOW_WIDTH),
            showlegend=False,
        )
        trace2 = go.Scatter(
            x=blocks,
            y=liquidities,
            name="active liquidity",
            hoverinfo="x+text",
            hovertext=[f"<b>Liquidity</b><br>{liquidity}" for liquidity in liquidities],
            mode="lines",
            line=dict(color=COLORS[1], width=1),
        )
        glow_trace2 = go.Scatter(
            x=blocks,
            y=liquidities,
            hoverinfo="skip",
            mode="lines",
            line=dict(color=GLOW_COLOR, width=GLOW_WIDTH),
            showlegend=False,
        )
        fig.add_trace(trace1, secondary_y=False)
        fig.add_trace(trace2, secondary_y=True)
        fig.add_trace(glow_trace1, secondary_y=False)
        fig.add_trace(glow_trace2, secondary_y=True)
        fig.update_layout(
            yaxis=dict(
                title=f"Price [{variables.data.params.pool_info[0].token0}/{variables.data.params.pool_info[0].token1}]"
            ),
            yaxis2=dict(title="Active Liquidity"),
        )
    fig.update_layout(
        xaxis=dict(title="Block", spikemode="across"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x",
    )
    fig.update_layout(font_family="Quicksand")
    fig = _add_bookmarks_to_fig(fig)
    fig.update_layout(xaxis=dict(tickformat="d"))
    return fig


def aaveV3_graph(agent_name: str):
    """Create the AaveV3 user data graph."""
    iagent = variables.data.params.agents.index(agent_name)
    blocks = sorted(list(variables.data.blockdata.keys()))
    debt = [
        variables.data.blockdata[block].agentdata[iagent].aaveV3data.totalDebt
        if variables.data.blockdata[block].agentdata[iagent].aaveV3data is not None
        else 0
        for block in blocks
    ]
    availableBorrows = [
        variables.data.blockdata[block].agentdata[iagent].aaveV3data.availableBorrows
        for block in blocks
    ]
    totalCollateral = [
        variables.data.blockdata[block].agentdata[iagent].aaveV3data.totalCollateral
        for block in blocks
    ]

    trace_debt = go.Scatter(
        x=blocks,
        y=debt,
        stackgroup="one",
        line=dict(color=COLORS[3], width=1),
        name="debt",
    )
    trace_availableBorrows = go.Scatter(
        x=blocks,
        y=availableBorrows,
        stackgroup="one",
        line=dict(color=COLORS[1], width=1),
        name="available borrows",
    )

    trace_collateral = go.Scatter(
        x=blocks,
        y=totalCollateral,
        stackgroup=None,
        line=dict(color=COLORS[0], width=1),
        name="total collateral",
    )
    trace_collateral_glow = go.Scatter(
        x=blocks,
        y=totalCollateral,
        stackgroup=None,
        line=dict(color=GLOW_COLOR, width=GLOW_WIDTH),
        showlegend=False,
        hoverinfo="skip",
    )

    fig = go.Figure(
        data=[
            trace_debt,
            trace_availableBorrows,
            trace_collateral_glow,
            trace_collateral,
        ]
    )
    fig.update_layout(
        xaxis=dict(title="Block", spikemode="across"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="amount (expressed in USDC)"),
        hovermode="x",
    )
    fig.update_layout(font_family="Quicksand")
    return fig


def aaveV3_graph_prices(token0, token1):
    """Greate price graph for AaveV3 tab."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    blocks = sorted(list(variables.data.blockdata.keys()))
    if len(blocks) == 0:
        return _empty_graph()

    if token0 is not None:
        y0 = [
            variables.data.blockdata[block].aaveV3_pooldata.token_prices[token0]
            if token0 in variables.data.blockdata[block].aaveV3_pooldata.token_prices
            else None
            for block in blocks
        ]
        trace0 = go.Scatter(
            x=blocks,
            y=y0,
            stackgroup=None,
            line=dict(color=COLORS[0], width=1),
            name=token0,
        )
        trace0_glow = go.Scatter(
            x=blocks,
            y=y0,
            stackgroup=None,
            line=dict(color=GLOW_COLOR, width=GLOW_WIDTH),
            showlegend=False,
            hoverinfo="skip",
        )
        fig.add_trace(trace0_glow)
        fig.add_trace(trace0)

    if token1 is not None:
        y1 = [
            variables.data.blockdata[block].aaveV3_pooldata.token_prices[token1]
            if token1 in variables.data.blockdata[block].aaveV3_pooldata.token_prices
            else None
            for block in blocks
        ]
        trace1 = go.Scatter(
            x=blocks,
            y=y1,
            stackgroup=None,
            line=dict(color=COLORS[1], width=1),
            name=token1,
        )
        trace1_glow = go.Scatter(
            x=blocks,
            y=y1,
            stackgroup=None,
            line=dict(color=GLOW_COLOR, width=GLOW_WIDTH),
            showlegend=False,
            hoverinfo="skip",
        )
        fig.add_trace(trace1_glow, secondary_y=True)
        fig.add_trace(trace1, secondary_y=True)

    fig.update_layout(
        xaxis=dict(title="Block", spikemode="across"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="token price in base_currency"),
        hovermode="x",
    )
    fig.update_layout(font_family="Quicksand")
    return fig
