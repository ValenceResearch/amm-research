"""Callbacks for the dashboard."""

import dash_bootstrap_components as dbc
import plotly.io as pio
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State

from dojo.vis import variables

pio.templates.default = "plotly_dark"


_progress_bar = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Progress(
                            label="PROGRESS",
                            value=50,
                            id="progress-bar",
                            color="linear-gradient(90deg, rgba(2,0,36,1) 0%, rgba(9,9,121,1) 35%, rgba(0,212,255,1) 100%)",
                            # style={'background-color': '#ff0000'}
                        )
                    ],
                    style={"padding": "0px 0px 0px"},
                    width=11,
                ),
                dbc.Col(
                    [
                        dbc.Button(
                            "⏸",
                            color="secondary",
                            className="mybutton",
                            n_clicks=0,
                            id="button-play-pause",
                        ),
                    ],
                    style={"text-align": "center", "padding": "0px 0px 0px"},
                    width=1,
                ),
            ]
        )
        # dbc.
    ],
    className="global-nav-card",
)


def _layout(mode):
    return html.Div(
        [
            dbc.Container(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.H6("Start Date"),
                                    dcc.Markdown(
                                        f"{variables.data.params.start_date}",
                                        id="info-start-date",
                                    ),
                                ]
                            ),
                            dbc.Col(
                                [
                                    html.H6("End Date"),
                                    dcc.Markdown(
                                        f"{variables.data.params.end_date}",
                                        id="info-end-date",
                                    ),
                                ],
                                style={"display": "None"} if mode == "live" else None,
                            ),
                            dbc.Col(
                                [
                                    html.H6("Environments"),
                                    dbc.Col(
                                        children=[
                                            ", ".join(
                                                variables.data.params.environments
                                            )
                                            if variables.data.params.environments
                                            else ""
                                        ],
                                        id="info-environments",
                                    ),
                                ]
                            ),
                            dbc.Col(
                                [
                                    html.H6("Agents"),
                                    html.Div(
                                        children=[],
                                        id="info-num-agents",
                                        # style={'line-height': '4px'}
                                    ),
                                ]
                            ),
                        ],
                        className="global-nav-card",
                    )
                ]
            ),
            _progress_bar,
        ]
    )


def _interactions(app: Dash, mode):
    """Helper function."""

    @app.callback(
        [
            Output("interval-component", "interval"),
            Output("button-play-pause", "children"),
        ],
        Input("button-play-pause", "n_clicks"),
        State("interval-component", "interval"),
        prevent_initial_call=True,
    )
    def play_pause(n_clicks, state):
        if state is None:
            return 2000, "⏸"
        else:
            return None, "⏵"

    @app.callback(
        [
            Output("progress-bar", "value"),
            Output("info-start-date", "children"),
            Output("info-end-date", "children"),
            Output("info-environments", "children"),
            Output("info-num-agents", "children"),
        ],
        [Input("interval-component", "n_intervals")],
        prevent_initial_call=True,
    )
    def update_graph(n):
        """Refresh the graph.

        :param n: Unused parameter.
        """
        progress = variables.data.params.progress_value

        return (
            progress,
            f"{variables.data.params.start_date}",
            f"{variables.data.params.end_date}",
            ", ".join(variables.data.params.environments),
            [
                html.P(f"{agent.name}-{agent.address}")
                if mode == "live"
                else html.P(f"{agent.name}")
                for agent in variables.data.params.agents
            ],
        )

    return app
