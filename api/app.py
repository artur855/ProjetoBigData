import joblib
import os
import cv2
import numpy as np
from flask import Flask, request
from tensorflow.keras.models import load_model

app = Flask(__name__)


api_path = os.path.dirname(__file__)
tradicional_path = os.path.join(api_path, 'tradicional.pkl')
cnn_path = os.path.join(api_path, 'cnn.h5')

tradicional = joblib.load(tradicional_path)
cnn = load_model(cnn_path)


def predict_tradicional(img):
    pred = tradicional.predict([img])[0]
    prob = tradicional.predict_proba([img])[0]
    index = np.where(tradicional.classes_ == pred)
    prob = prob[index][0]

    return pred, prob


def predict_cnn(img):
    classes = ['apple', 'beach', 'bench', 'bread', 'cloud']
    img = np.array([img, ])
    pred = cnn.predict_classes(img)[0]
    prob = cnn.predict(img)[0]
    prob = float(prob[pred])
    pred = classes[pred]
    return pred, prob


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/which_draw', methods=['POST'])
def whichdraw():
    if 'file' in request.files:
        file = request.files['file'].read()
        npimg = np.frombuffer(file, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (50, 50))
        img = img.flatten()
        pred_tradicional, prob_tradicional = predict_tradicional(img)
        pred_cnn, prob_cnn = predict_cnn(img)
        return {
            'tradicional': pred_tradicional,
            'tradicional_prob': prob_tradicional,
            'cnn': pred_cnn,
            'cnn_prob': prob_cnn
        }


if __name__ == "__main__":
    app.run(host='0.0.0.0')
