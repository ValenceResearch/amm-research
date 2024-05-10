"""Sending data to the Dashboard."""
import concurrent.futures
import json

import jsons
import requests

from dojo.vis.variables import BlockData, Params


class Plotter:
    """Sending data to the dashboard for plotting."""

    def __init__(self, port: int = 8051):
        """Initialize a new plotter instance.

        :param port: The port on which the dashboard is running.
        """
        self.port = port
        self.address = "http://0.0.0.0"
        self.headers = {"Content-Type": "application/json"}

    def update_blockdata(self, block: int, blockdata: BlockData):
        """Update date for a particular simulation block."""
        URL = f"{self.address}:{self.port}/blockdata"
        payload = {"block": block, "data": jsons.dump(blockdata)}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(self.send_post, *["POST", URL, self.headers, payload])

    def send_post(self, method: str, url: str, headers: dict, payload: str):
        """Helper function."""
        requests.request(method, url, headers=headers, data=json.dumps(payload))

    def send_progress(self, progress: int):
        """Send progress to the dashboard."""
        URL = f"{self.address}:{self.port}/progress"
        payload = {"progress": progress}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(self.send_post, *["POST", URL, self.headers, payload])

    def send_pid(self, pid: int):
        """Send progress to the dashboard."""
        URL = f"{self.address}:{self.port}/pid"
        payload = {"pid": pid}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(self.send_post, *["POST", URL, self.headers, payload])

    def send_params(self, params: Params):
        """Send info to the dashboard."""
        URL = f"{self.address}:{self.port}/params"
        payload = jsons.dumps(params)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(self.send_post, *["POST", URL, self.headers, payload])

    def send_param(self, key, value):
        """Send info to the dashboard."""
        URL = f"{self.address}:{self.port}/param"
        payload = jsons.dumps({key: value})
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(self.send_post, *["POST", URL, self.headers, payload])

    def send_info(self, params: Params):
        """Send info to the dashboard."""
        URL = f"{self.address}:{self.port}/info"
        payload = json.dumps(params)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(self.send_post, *["POST", URL, self.headers, payload])
