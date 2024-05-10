"""Callbacks for the dashboard."""

import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import plotly.io as pio
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State

from dojo.vis import variables
from dojo.vis.graphs import _empty_graph, positions_graph, price_graph

pio.templates.default = "plotly_dark"


def _layout():
    return dbc.Tab(
        children=[
            html.P(" "),
            html.H6("Select a pool..."),
            dbc.Col(
                dmc.Select(
                    # label="Select pool",
                    placeholder="...",
                    value=None,
                    data=[
                        {"value": f"{pool_info.name}", "label": f"{pool_info.name}"}
                        for pool_info in variables.data.params.pool_info
                    ],
                    id="pool-select",
                ),
                width=6,
            ),
            dcc.Markdown("---"),
            html.H6("Positions"),
            html.P(
                "This graph shows your positions in both tokens over time.",
                className="mt-2",
            ),
            dcc.Graph(id="positions-graph", figure=_empty_graph()),
            html.P(" "),
            dcc.Markdown("---"),
            html.H6("Pool data"),
            html.P(
                "This graph shows the token price as well as the total liquidity in the pool.",
                className="mt-2",
            ),
            dcc.Graph(id="graph-price", figure=_empty_graph()),
        ],
        label="UniswapV3",
        label_style={"color": "#FF67C3"},
        active_label_style={
            "color": "#FF67C3",
            "background-color": "rgba(255,255,255,0.1)",
        },
    )


def _interactions(app: Dash):
    """Helper function."""

    @app.callback(
        Output("bookmarks-table", "data", allow_duplicate=True),
        Input("positions-graph", "clickData"),
        prevent_initial_call=True,
    )
    def create_bookmark_positions(clickData):
        if clickData:
            bm = variables.Bookmark(name="name", block=int(clickData["points"][0]["x"]))
            variables.data.bookmarks += [bm]

        data = [
            {"number": f"{i}", "name": f"{bm.name}"}
            for i, bm in enumerate(variables.data.bookmarks)
        ]
        return data

    @app.callback(
        Output("bookmarks-table", "data", allow_duplicate=True),
        Input("graph-price", "clickData"),
        prevent_initial_call="initial_duplicate",
    )
    def create_bookmark_price(clickData):
        if clickData:
            bm = variables.Bookmark(name="name", block=int(clickData["points"][0]["x"]))
            variables.data.bookmarks += [bm]

        data = [
            {"number": f"{i}", "name": f"{bm.name}"}
            for i, bm in enumerate(variables.data.bookmarks)
        ]
        return data

    @app.callback(
        [
            Output("positions-graph", "figure"),
            Output("pool-select", "data"),
        ],
        [Input("interval-component", "n_intervals")],
        [State("pool-select", "value")],
        prevent_initial_call=True,
    )
    def update_graph(n, selected_pool):
        """Refresh the graph.

        :param n: Unused parameter.
        """
        if selected_pool is None:
            fig_price = _empty_graph()
        else:
            fig_price = positions_graph(selected_pool)

        return (
            fig_price,
            [
                {"value": f"{pool_info.name}", "label": f"{pool_info.name}"}
                for pool_info in variables.data.params.pool_info
            ],
        )

    @app.callback(
        Output("graph-price", "figure"),
        [Input("interval-component", "n_intervals")],
        [State("pool-select", "value")],
        prevent_initial_call=True,
    )
    def update(n, selected_pool):
        """Refresh the graph.

        :param n: Unused parameter.
        """
        if selected_pool is None:
            return _empty_graph()
        return price_graph(selected_pool)

    return app
