"""Callbacks for the dashboard."""

import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import plotly.io as pio
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State

from dojo.vis import variables
from dojo.vis.graphs import _empty_graph, aaveV3_graph, aaveV3_graph_prices

pio.templates.default = "plotly_dark"


def _layout():
    return dbc.Tab(
        children=[
            html.P(" "),
            html.P(" "),
            html.H6("Agent data"),
            html.P(
                "This graph shows the agent positions in base currency over time. If the  total debt(red) ever exceeds the total available borrows (blue), AAVE liquidates the collateral.",
                className="mt-2",
            ),
            dbc.Row(
                dbc.Col(
                    children=dmc.Select(
                        label="Select agent",
                        placeholder="...",
                        value=None,
                        data=[
                            {"value": iagent, "label": f"agent {iagent}"}
                            for iagent in range(len(variables.data.params.agents))
                        ],
                        id="agent-select-aaveV3",
                    ),
                    width=6,
                )
            ),
            dcc.Graph(id="graph-aaveV3", figure=_empty_graph()),
            html.Hr(),
            html.H6("Pool data"),
            html.P(
                "This graph shows selected token prices over time, expressed in base currency .",
                className="mt-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        children=dmc.Select(
                            label="Select token",
                            placeholder="...",
                            value=None,
                            data=[],
                            id="aave-token-select0",
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        children=dmc.Select(
                            label="Select token",
                            placeholder="...",
                            value=None,
                            data=[],
                            id="aave-token-select1",
                        ),
                        width=6,
                    ),
                ]
            ),
            dcc.Graph(id="graph-aaveV3-prices", figure=_empty_graph()),
        ],
        label="AaveV3",
        label_style={"color": "#FF67C3"},
        active_label_style={
            "color": "#FF67C3",
            "background-color": "rgba(255,255,255,0.1)",
        },
    )


def _interactions(app: Dash):
    """Helper function."""

    @app.callback(
        [Output("aave-token-select0", "data"), Output("aave-token-select1", "data")],
        Input("interval-component", "n_intervals"),
        [State("aave-token-select0", "data"), State("aave-token-select1", "data")],
    )
    def update_agent_select(n, data0, data1):
        if True:  # n < 10:
            blocks = sorted(list(variables.data.blockdata.keys()))
            if len(blocks) == 0:
                return None, None
            tokens = variables.data.blockdata[
                blocks[0]
            ].aaveV3_pooldata.token_prices.keys()
            data = [{"value": token, "label": token} for token in tokens]
            return data, data
        # return data0, data1

    @app.callback(
        Output("graph-aaveV3-prices", "figure"),
        [
            Input("interval-component", "n_intervals"),
            Input("aave-token-select0", "value"),
            Input("aave-token-select1", "value"),
        ],
    )
    def update_price_graph(n, token0, token1):
        """Refresh the graph.

        :param n: Unused parameter.
        """
        fig_prices = aaveV3_graph_prices(token0, token1)
        return fig_prices

    @app.callback(
        [Output("graph-aaveV3", "figure"), Output("agent-select-aaveV3", "data")],
        [
            Input("interval-component", "n_intervals"),
            Input("agent-select-aaveV3", "value"),
        ],
        prevent_initial_call=True,
    )
    def update(n, selected_agent):
        """Refresh the graph.

        :param n: Unused parameter.
        """
        if selected_agent is None:
            fig = _empty_graph()
        else:
            fig = aaveV3_graph(selected_agent)
        return fig, [
            {"value": agent, "label": agent} for agent in variables.data.params.agents
        ]

    return app
