from flask import Flask, request, jsonify

def create_app():
    app = Flask(__name__)

    @app.route('/server', methods=['POST'])
    def server():
        data = request.get_json()
        return jsonify({"you_sent": data})

    return app

def main():
    app = create_app()
    app.run(host='0.0.0.0', port=10000, debug=True)


if __name__ == '__main__':
    main()