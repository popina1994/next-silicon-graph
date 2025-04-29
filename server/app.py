from flask import Flask, request, jsonify
import pygraphviz as pgv
import logging

def create_app():
    app = Flask(__name__)

    def generate_error(str_error: str):
        return jsonify({"Error": str_error})

    @app.route('/server', methods=['POST'])
    def server():
        data_json = request.get_json()
        data_str = str(data_json)
        if not 'e1' in data_json:
            str_error = "No entry node is specified"
            app.logger.error(str_error + data_str)
            error_json = generate_error(str_error)
            return error_json

        if not 'h' in data_json:
            str_error = "No node in the graph is specified"
            app.logger.error(str_error + data_str)
            error_json = generate_error(str_error)
            return error_json

        if not 'graph' in data_json:
            str_error = "No graph is specified"
            app.logger.error(str_error + data_str)
            error_json = generate_error(str_error)
            return error_json

        try:
            graph = pgv.AGraph(string=data_json['graph'])  # Parsing the DOT string
            print("Graph parsed successfully.")
        except pgv.AGraphParseError as e:
            str_error = "Error in parsing dot string"
            app.logger.error(str_error + data_str)
            error_json = generate_error(str_error)
            return error_json
        except pgv.AGraphError as e:
            str_error = "Error in parsing dot string"
            app.logger.error(str_error + data_str)
            error_json = generate_error(str_error)
            return error_json

        return jsonify({"you_sent": data_json})

    return app

def init_logging(app, log_level):
    LOG_FILE_NAME = 'app.log'
    file_handler = logging.FileHandler(LOG_FILE_NAME)
    file_handler.setLevel(log_level)
    log_format_str = '%(asctime)s - %(levelname)s - %(filename)s - line %(lineno)d - %(message)s'

    formatter = logging.Formatter(log_format_str)
    file_handler.setFormatter(formatter)

    logging.basicConfig(level=log_level, format=log_format_str)

    app.logger.addHandler(file_handler)
    app.logger.setLevel(log_level)


def main():
    app = create_app()
    log_level = logging.DEBUG
    HOST_IP = '0.0.0.0'
    PORT = 10000
    init_logging(app, log_level)

    app.run(host=HOST_IP, port=PORT, debug=True)


if __name__ == '__main__':
    main()