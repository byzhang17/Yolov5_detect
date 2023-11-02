from flask import Flask, jsonify, request

app = Flask(__name__)


class DetectAPI(object):
    def __init__(self):
        self.i = 1

    def run1(self):
        print("hello world")


detect_test = DetectAPI()


@app.route('/test', methods=['POST'])
def printtest():
    detect_test.run1()
    return jsonify({"test": str("hello")})


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=3333)
