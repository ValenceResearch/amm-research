"""Dashboard to visualize simulation."""

import jsons
from flask import request
from flask.app import Flask

from dojo.vis import variables


def register_api(server: Flask) -> Flask:
    """Helper function."""

    @server.route("/info", methods=["POST"])
    def update_info():
        """API endpoint."""
        post_data = request.get_json()
        if "num_agents" in post_data:
            variables.data.params.num_agents = post_data["num_agents"]
        if "start_date" in post_data:
            variables.data.params.start_date = post_data["start_date"]
        if "end_date" in post_data:
            variables.data.params.end_date = post_data["end_date"]
        if "pool" in post_data:
            variables.data.params.pool = post_data["pool"]
        if "token0" in post_data:
            variables.data.params.token0 = post_data["token0"]
        if "token1" in post_data:
            variables.data.params.token1 = post_data["token1"]
        if "pool_fee" in post_data:
            variables.data.params.pool_fee = float(post_data["pool_fee"])
        if "num_agents" in post_data:
            variables.data.params.num_agents = int(post_data["num_agents"])
        if "signal_names" in post_data:
            variables.data.params.signal_names.update(set(post_data["signal_names"]))

        return "Info updated successfully"

    @server.route("/pid", methods=["POST"])
    def update_pid():
        """API endpoint."""
        post_data = request.get_json()
        variables.pid = post_data["pid"]
        return "Success"

    @server.route("/params", methods=["POST"])
    def update_params():
        post_data = request.get_json()
        params = jsons.loads(post_data, variables.Params)
        variables.data.params = params
        return "Updated"

    @server.route("/param", methods=["POST"])
    def update_param():
        post_data = jsons.loads(request.get_json())
        if "signal_names" in post_data:
            variables.data.params.signal_names = post_data["signal_names"]
            return "Updated"
        return "No Updates"

    @server.route("/progress", methods=["POST"])
    def update_progress():
        """API endpoint."""
        new_data = request.get_json()
        variables.data.params.progress_value = new_data["progress"]
        return "Progress updates successfully"

    @server.route("/blockdata", methods=["POST"])
    def update_blockdata():
        post_data = request.get_json()
        blockdata = jsons.load(post_data["data"], variables.BlockData)
        block = int(post_data["block"])

        variables.data.blockdata[block] = blockdata

        return "Blockdata updated successfully"

    return server
