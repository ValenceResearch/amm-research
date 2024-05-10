"""Callbacks for the dashboard."""
import base64
import os
from enum import Enum
from signal import SIGTERM

import dash_bootstrap_components as dbc
import jsons
import plotly.io as pio
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

from dojo.vis import variables

ActionType = Enum("ActionType", ["Trade", "Quote"])


pio.templates.default = "plotly_dark"


_file_upload_success_modal = dbc.Modal(
    [
        dbc.ModalHeader(
            dbc.ModalTitle("Choose the .json data file"),
            style={"background": "#1e1c2d"},
        ),
        dbc.ModalBody(
            [
                html.P(
                    "Please be aware that data files created in older version of dojo might not load correctly."
                ),
                dcc.Upload(
                    id="upload-data",
                    children=html.Div(["Drag and Drop or ", html.A("Select File")]),
                    style={
                        "width": "100%",
                        "height": "60px",
                        "lineHeight": "60px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",
                        "margin": "10px",
                    },
                    # Allow multiple files to be uploaded
                    multiple=False,
                ),
            ],
            style={"background": "#1e1c2d"},
        ),
    ],
    id="file_upload_success_modal",
    is_open=False,
)

_buttons_bar = dbc.Row(
    [
        dbc.Col(
            dbc.Button(
                "Load",
                color="secondary",
                className="mybutton",
                n_clicks=0,
                id="button-load",
            ),
            width="auto",
        ),
        dbc.Col(
            dbc.Button(
                "Save",
                color="secondary",
                className="mybutton",
                n_clicks=0,
                id="button-save",
            ),
            width="auto",
        ),
        dbc.Col(
            dbc.Button(
                "Reset",
                color="secondary",
                className="mybutton",
                n_clicks=0,
                id="button-reset",
            ),
            width="auto",
        ),
        dbc.Col(
            dbc.Button(
                "KILL",
                color="secondary",
                className="killbutton",
                n_clicks=0,
                id="button-kill",
            ),
            width="auto",
        ),
    ],
    className="g-0 ms-auto flex-nowrap mt-3 mt-md-0",
    # align="right",
)


def _navbar(mode):
    return dbc.Container(
        [
            dbc.Navbar(
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.A(
                                            # Use row and col to control vertical alignment of logo / brand
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        html.Img(
                                                            src="/assets/logo.svg",
                                                            height="50px",
                                                        ),
                                                        className="no_borders",
                                                    ),
                                                    dbc.Col(
                                                        dbc.NavbarBrand(
                                                            "Dojo Execution Dashboard"
                                                            if mode == "live"
                                                            else "Dojo Simulation Dashboard",
                                                            className="display-1 ms-2 h4",
                                                            style={
                                                                # "color": "#ff0000",
                                                                "font-size": "2rem",
                                                                "font-weight": "bold",
                                                            },
                                                        ),
                                                        className="no_borders h-4",
                                                    ),
                                                ],
                                                align="center",
                                                className="g-0",
                                            ),
                                            href="https://dojo.compasslabs.ai",
                                            style={"textDecoration": "none"},
                                        )
                                    ],
                                    width="auto",
                                ),
                                dbc.Col(
                                    [
                                        dbc.NavbarToggler(
                                            id="navbar-toggler", n_clicks=0
                                        ),
                                        dbc.Collapse(
                                            _buttons_bar,
                                            id="navbar-collapse",
                                            is_open=False,
                                            navbar=True,
                                        ),
                                    ],
                                    width="auto",
                                ),
                                # dcc.Markdown("---")
                            ],
                            justify="between",
                        ),
                        # dcc.Markdown("---"),
                    ]
                ),
                # color="dark",
                # className="global-nav-card",
                dark=True,
                sticky="top"
                # style={"border-left": "None", "border-right": "None", "margin-left": "unset"},
            )
        ]
    )


def _layout(mode):
    return html.Div(
        [
            dcc.Download(id="download-csv"),
            dcc.Upload(id="upload-csv"),
            _file_upload_success_modal,
            dcc.Store(id="data"),
            _navbar(mode),
        ]
    )


def _interactions(app: Dash, demo_file: str = None):
    """Helper function."""

    @app.callback(
        Output("button-kill", "children"),
        Input("button-kill", "n_clicks"),
        prevent_initial_call=True,
    )
    def kill_live_runner(n):
        if variables.pid is not None:
            os.kill(variables.pid, SIGTERM)
        return "KILL"

    @app.callback(
        Output("download-csv", "data"),
        Input("button-save", "n_clicks"),
        prevent_initial_call=True,
    )
    def save_data(clickData):
        return dict(
            content=jsons.dumps(variables.data),
            filename="dojo.json",
        )

    @app.callback(
        Output("file_upload_success_modal", "is_open", allow_duplicate=True),
        Input("button-load", "n_clicks"),
        prevent_initial_call=True,
    )
    def open_modal(n_clicks):
        return True

    @app.callback(
        Output("file_upload_success_modal", "is_open", allow_duplicate=True),
        Input("upload-data", "contents"),
        prevent_initial_call=True,
    )
    def update_output(contents):
        json_string = contents.split(",")[1]
        decoded_data = base64.b64decode(json_string)
        variables.data = jsons.loads(decoded_data, variables.Data)
        return False

    @app.callback(
        Output("interval-component", "n_intervals"),
        Input("button-reset", "n_clicks"),
    )
    def reset_dashboard(a):
        if variables.is_demo:
            variables._from_file(demo_file)
        else:
            variables.reset()
        return 0

    return app
