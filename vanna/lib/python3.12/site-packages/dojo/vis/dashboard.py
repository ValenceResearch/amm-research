"""Dashboard to visualize simulation."""
import argparse
import logging
import os
import webbrowser
from enum import Enum
from threading import Timer
from typing import List, Literal

import dash
import dash_bootstrap_components as dbc
from dash import dcc
from flask import Flask
from waitress import serve

from dojo.vis import variables
from dojo.vis.dashboard_api import register_api
from dojo.vis.interactions import aaveV3, bookmarks, explore, info, io, rewards, uniswap

ActionType = Enum("ActionType", ["Trade", "Quote"])


def _kill_dashboard(port):
    def _f():
        os.system(f"kill $(lsof -t -i:{port})")

    return _f


def run_app(
    port: int = 8051,
    mode: Literal["prod", "dev", "demo", "live"] = "prod",
    envs: List[Literal["uniswap", "aave"]] = ["uniswap"],
    demo_file: str = "example_uniswap.json",
    jupyter=False,
):
    """Start the dashboard.

    :param mode: 'prod' for production.
    :param port: The port on which the dashboard is running. The plotter must send data
        to this port.
    :param jupyter: Set this to true if you want to run the dashboard inline within a
        Jupyter notebook.
    :raises ValueError: Temporarily removed Jupyter support.
    """

    def open_browser():
        webbrowser.open_new("http://0.0.0.0:{}".format(port))

    # global variables
    if mode == "demo":
        variables.is_demo = True

    # Initialize the Flask application
    server = Flask(__name__)

    server = register_api(server)

    # Initialize the Dash application
    if jupyter is True:
        raise ValueError("Jupyter support has been removed(for now).")
    else:
        app = dash.Dash(
            __name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP]
        )
    app.title = "dojo"
    app._favicon = "logo.svg"

    if "aave" in envs:
        layout = [
            rewards._layout(),
            aaveV3._layout(),
            explore._layout(),
            bookmarks._layout(),
        ]
    elif "uniswap" in envs:
        layout = [
            rewards._layout(),
            uniswap._layout(),
            explore._layout(),
            bookmarks._layout(),
        ]

    # Create the layout of the dashboard
    app.layout = dbc.Container(
        [
            io._layout(mode),
            info._layout(mode),
            dbc.Container(
                [
                    dbc.Tabs(
                        layout,
                        id="tabs",
                        active_tab="tab-1",
                    )
                ],
                className="global-nav-card",
            ),
            dcc.Interval(
                id="interval-component",
                interval=2000,  # Refresh interval in milliseconds
                n_intervals=0,
            ),
            dash.html.Footer("© 2023 - CompassLabs", className="center"),
        ],
        fluid=True,
        # style={"margin-left": "300px"},
    )

    app = info._interactions(app, mode)
    app = bookmarks._interactions(app)
    app = rewards._interactions(app)
    if "uniswap" in envs:
        app = uniswap._interactions(app)
    if "aave" in envs:
        app = aaveV3._interactions(app)
    app = explore._interactions(app)
    app = io._interactions(app, demo_file=demo_file)

    if mode != "demo" and mode != "dev":
        Timer(2, open_browser).start()
    if jupyter is True:
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        app.run_server(debug=False, mode="inline", port=port)
    else:
        if mode == "dev" or mode == "demo" or mode == "live":
            app.run_server(debug=True, port=port)
        else:
            host = "0.0.0.0"
            serve(app.server, host=host, port=port)


if __name__ == "__main__":
    # Run the application

    parser = argparse.ArgumentParser(description="Dojo dashboard.")
    parser.add_argument(
        "--mode",
        choices=["prod", "dev", "demo"],
        default="prod",
        help="Specify `prod` for regular use.",
    )
    parser.add_argument(
        "--env",
        choices=["uniswap", "aave"],
        default="uniswap",
        help="Specify the exchange involved in the simulation.",
    )
    parser.add_argument(
        "--demo_file",
        nargs="?",
        default=None,
        help="Demo file location. Only used in `demo` mode.",
    )
    # Add an integer parameter
    parser.add_argument(
        "--port", type=int, default=8051, help="Specify the port number"
    )
    args = parser.parse_args()

    run_app(port=args.port, mode="dev", envs=["uniswap"], demo_file=args.demo_file)
