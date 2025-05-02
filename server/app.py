"""
Module providing a support for the handler of the REST requests.
"""
import logging
from flask import Flask, request, jsonify
import pygraphviz as pgv
from logic.graph import DataFlowGraph
from logic.graph import NodeDoesNotExist

def create_app():
    """
    Creates a Flask app API that is used to launch the app on the specific host and port.
    """
    app = Flask(__name__)

    def generate_error(str_error: str):
        return jsonify({"Error": str_error})

    @app.route('/server', methods=['POST'])
    def server():
        data_json = request.get_json()
        data_str = str(data_json)
        if not 'e1' in data_json:
            str_error = "No entry node is specified"
            app.logger.error("%s, %s", str_error, data_str)
            error_json = generate_error(str_error)
            return error_json

        if not 'h' in data_json:
            str_error = "No desination node in the graph is specified"
            app.logger.error("%s, %s", str_error, data_str)
            error_json = generate_error(str_error)
            return error_json

        if not 'graph' in data_json:
            str_error = "No graph is specified"
            app.logger.error("%s, %s", str_error, data_str)
            error_json = generate_error(str_error)
            return error_json

        try:
            pg_graph = pgv.AGraph(string=data_json['graph'])
            start_node_name = data_json['e1']
            reach_node_name = data_json['h']
            print("Graph parsed successfully.")
            # Extract nodes
            node_names = pg_graph.nodes()

            # Extract edges
            edge_names = pg_graph.edges()
            data_flow_graph = DataFlowGraph(node_names, edge_names, start_node_name, app.logger)
            dominate_node_names = data_flow_graph.get_dominate_nodes(reach_node_name)

            return jsonify({"come_before_node": dominate_node_names})

        except NodeDoesNotExist as e:
            str_error =f"{e}"
            app.logger.error("%s, %s", str_error, data_str)
            error_json = generate_error(str_error)
            return error_json
        except (OSError, ValueError) as _:
            str_error = "Error in parsing dot string"
            app.logger.error("%s, %s", str_error, data_str)
            error_json = generate_error(str_error)
            return error_json

    return app


def init_logging(app, log_level, log_file_name):
    """
    Initializes logging including log_levle, file format, and
    redirection to the file log_file_name
    """
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setLevel(log_level)
    log_format_str = '%(asctime)s - %(levelname)s - %(filename)s - line %(lineno)d - %(message)s'

    formatter = logging.Formatter(log_format_str)
    file_handler.setFormatter(formatter)

    logging.basicConfig(level=log_level, format=log_format_str)

    app.logger.addHandler(file_handler)
    app.logger.setLevel(log_level)


def main():
    """
    Entry point to the main function of the server
    """
    app = create_app()
    log_level = logging.DEBUG
    log_file_name = 'app.log'
    host_ip = '0.0.0.0'
    port = 10000
    init_logging(app, log_level, log_file_name)

    app.run(host=host_ip, port=port, debug=False)


if __name__ == '__main__':
    main()
