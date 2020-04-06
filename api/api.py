from flask import Flask
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/which_draw')
def whichdraw():
    return {1: 1}


if __name__ == "__main__":
    app.run(host='0.0.0.0')
