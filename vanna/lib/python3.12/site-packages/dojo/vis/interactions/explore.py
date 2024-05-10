"""Callbacks for the dashboard."""
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import plotly.io as pio
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

from dojo.vis import variables
from dojo.vis.graphs import _empty_graph, custom_figure

pio.templates.default = "plotly_dark"


def _layout():
    return dbc.Tab(
        children=[
            html.P(" "),
            html.P(
                "Use the this tool to explore the simulation. You can plot any 2 quantities against each other over time.",
                className="mt-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dmc.Select(
                            label="Left quantity",
                            placeholder="Select left quantity",
                            value="",
                            data=[],
                            id="explore-select-0",
                        ),
                        width=5,
                    ),
                    dbc.Col(
                        dmc.Select(
                            label="Right category",
                            placeholder="Select right quantity",
                            value="",
                            data=[],
                            id="explore-select-1",
                        ),
                        width=5,
                    ),
                    dbc.Col(
                        children=[
                            html.P(" "),
                            dbc.Checkbox(
                                label="same axis", name="name", id="checkbox-same-axis"
                            ),
                        ],
                        width=2,
                    ),
                ]
            ),
            dcc.Graph(id="custom-graph", figure=_empty_graph()),
        ],
        label="Explore",
        label_style={"color": "#FF67C3"},
        active_label_style={
            "color": "#FF67C3",
            "background-color": "rgba(255,255,255,0.1)",
        },
        style=dict(background="transparent"),
    )


def _interactions(app: Dash):
    """Helper function."""

    @app.callback(
        Output("custom-graph", "figure"),
        [
            Input("explore-select-0", "value"),
            Input("explore-select-1", "value"),
            Input("checkbox-same-axis", "value"),
            Input("interval-component", "n_intervals"),
        ],
    )
    def custom_select(v1, v2, same_axis, n_interval):
        return custom_figure(v1, v2, not same_axis)

    @app.callback(
        [
            Output("explore-select-0", "data"),
            Output("explore-select-1", "data"),
        ],
        [Input("interval-component", "n_intervals")],
        prevent_initial_call=True,
    )
    def update_explore_tab(n_interval):

        options = []
        for iagent, agent in enumerate(variables.data.params.agents):
            options.append(
                {
                    "value": f"agent-{agent.name}-reward",
                    "label": "Reward",
                    "group": f"Agent-{agent.name}",
                }
            )
        for pool_info in variables.data.params.pool_info:
            options.append(
                {
                    "value": f"pool-{pool_info.name}-liquidity",
                    "label": "Liquidity",
                    "group": f"Pool-{pool_info.name}",
                }
            )
            options.append(
                {
                    "value": f"pool-{pool_info.name}-price",
                    "label": "Price",
                    "group": f"Pool-{pool_info.name}",
                }
            )
        for signal_name in variables.data.params.signal_names:
            options.append(
                {
                    "value": f"signal-{signal_name}",
                    "label": f"{signal_name}",
                    "group": "Signals",
                }
            )

        return (options, options)

    return app
