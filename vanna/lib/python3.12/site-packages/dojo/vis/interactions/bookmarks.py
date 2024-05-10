"""Callbacks for the dashboard."""
from enum import Enum

import dash_bootstrap_components as dbc
import plotly.io as pio
from dash import Dash, dash_table, html
from dash.dependencies import Input, Output

from dojo.vis import variables

ActionType = Enum("ActionType", ["Trade", "Quote"])


pio.templates.default = "plotly_dark"


_bookmarks_table = dash_table.DataTable(
    data=[
        {"number": f"{i}", "name": f"{bm.name}"}
        for i, bm in enumerate(variables.data.bookmarks)
    ],
    columns=[{"name": "number", "id": "number"}, {"name": "name", "id": "name"}],
    style_cell={"textAlign": "left"},
    style_data={
        "backgroundColor": "rgba(251, 250, 255, .05)",
        "border": "1px solid lightgrey",
        # 'textAlign': 'left'
    },
    style_header={
        "backgroundColor": "rgba(251, 250, 255, .15)",
        "color": "white",
        "fontWeight": "bold",
        "border": "1px solid lightgrey",
        # 'textAlign': 'center'
    },
    style_table={
        # 'border': '1px solid red',
        "borderRadius": "15px",
        # 'overflow': 'hidden'
    },
    id="bookmarks-table",
)


_create_bookmark_modal = dbc.Modal(
    [
        dbc.ModalHeader(
            dbc.ModalTitle("Data for selected block"),
            style={"background": "#1e1c2d"},
        ),
        dbc.ModalBody(
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Input(
                                id="input-bookmark-name",
                                placeholder="Name bookmark...",
                                type="text",
                            ),
                            dbc.Button(
                                "Save",
                                color="secondary",
                                className="mybutton",
                                n_clicks=0,
                                id="button-bookmark",
                            ),
                        ],
                        width="auto",
                    ),
                ],
                className="g-0 ms-auto flex-nowrap mt-3 mt-md-0",
                # align="right",
            ),
        ),
    ],
    id="inspector-modal",
    is_open=False,
    className="global-nav-card",
)


def _layout():
    return dbc.Tab(
        children=[html.P(" "), _bookmarks_table, _create_bookmark_modal],
        label="Bookmarks",
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
            Output("inspector-modal", "is_open"),
            Output("bookmarks-table", "data", allow_duplicate=True),
        ],
        Input("live-update-graph", "clickData"),
        prevent_initial_call=True,
    )
    def display_click_data(clickData):
        if clickData:
            bm = variables.Bookmark(name="name", block=int(clickData["points"][0]["x"]))
            variables.data.bookmarks += [bm]

        data = [
            {"number": f"{i}", "name": f"{bm.name}"}
            for i, bm in enumerate(variables.data.bookmarks)
        ]
        return False, data

    return app
