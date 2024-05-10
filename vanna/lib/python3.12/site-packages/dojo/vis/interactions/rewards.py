"""Callbacks for the dashboard."""

import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import plotly.io as pio
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State

from dojo.vis import variables
from dojo.vis.graphs import _empty_graph, reward_graph

pio.templates.default = "plotly_dark"


def _layout():
    return dbc.Tab(
        children=[
            html.P(" "),
            html.P(
                "This graph shows your agents reward over time",
                className="mt-2",
            ),
            dcc.Markdown("**The Solid line** shows the reward value."),
            dcc.Markdown(
                "**Dots** indicate blocks where your agent took actions. Hover over the dots to get more info."
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
                        id="agent-select",
                    ),
                    width=6,
                )
            ),
            dcc.Graph(id="live-update-graph", figure=_empty_graph()),
        ],
        label="Rewards",
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
        [
            Output("live-update-graph", "figure"),
            Output("agent-select", "data"),
        ],
        [Input("interval-component", "n_intervals")],
        [State("agent-select", "value")],
        prevent_initial_call=True,
    )
    def update_graph(n, selected_agent):
        """Refresh the graph.

        :param n: Unused parameter.
        """
        fig_rewards = reward_graph(selected_agent)
        return (
            fig_rewards,
            [
                {"value": agent.name, "label": agent.name}
                for agent in variables.data.params.agents
            ],
        )

    return app
