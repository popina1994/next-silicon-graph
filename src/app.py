from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, Flask!"

@app.route('/server', methods=['POST'])
def echo():
    data = request.get_json()
    return jsonify({"you_sent": data})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
