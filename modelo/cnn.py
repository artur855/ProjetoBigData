import json
import cv2
import numpy as np
from pprint import pprint
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils


def pre_process_image(img):
    process_img = cv2.resize(img, (200, 200))
    process_img = process_img[:, :, 0]
    return process_img


def get_blob(matrix):

    max_h = []
    max_w = []

    for stroke in matrix:
        max_w.append(max(stroke[0]))
        max_h.append(max(stroke[1]))

    h = max(max_h)
    w = max(max_w)

    image = Image.new('RGB', (w+5, h+5))
    draw = ImageDraw.Draw(image)

    for stroke in matrix:
        xy = list(zip(stroke[0], stroke[1]))
        draw.line(xy)
    return pre_process_image(np.array(image))
    # return cv2.resize(np.array(image), (25, 25))


with open('data_set\\full_simplified_cloud.ndjson', encoding='utf-8') as file:
    jsons = file.read().strip().split('\n')[:39373]
    jsons = [json.loads(j) for j in jsons]
    jsons = [_json['drawing'] for _json in jsons]
    jsons = [get_blob(_json) for _json in jsons]

jsons = np.array(jsons)
y = np.array([0]*len(jsons))

x_train, x_test, y_train, y_test = train_test_split(jsons, y, test_size=0.25)
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape(len(x_test), np.prod(x_test.shape[1:]))
y_train = np_utils.to_categorical(y_train, 5)
y_test = np_utils.to_categorical(y_test, 5)

model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=200*200))
model.add(Dropout(0.2))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=5, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
historico = model.fit(x_train, y_train, epochs=5,
                      validation_data=(x_test, y_test))
print(historico)
pred = model.predict(x_test)
test_matrix = [np.argmax(t) for t in y_test]
pred_matrix = [np.argmax(t) for t in pred]
confunsion = confusion_matrix(test_matrix, pred_matrix)
