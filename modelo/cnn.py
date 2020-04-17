import json
import cv2
import os
import numpy as np
from pprint import pprint
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from matplotlib import pyplot as plt


def pre_process_image(img):
    process_img = cv2.resize(img, (50, 50))
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


data = []
dataset_path = os.path.join(os.path.dirname(__file__), 'data_set')
for dataset_file in os.listdir(dataset_path):
    dataset_file_path = os.path.join(dataset_path, dataset_file)
    print('Processando: ' + dataset_file)
    with open(dataset_file_path) as file:
        jsons = file.read().strip().split('\n')
        jsons = [json.loads(j) for j in jsons]
        data += jsons
print('Gerando blobs')
x = [get_blob(_data['drawing']) for _data in data]
y = [_data['word'] for _data in data]
map_classes = {
    'apple': 0,
    'beach': 1,
    'bench': 2,
    'bread': 3,
    'cloud': 4
}
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_train = np.array(x_train)
x_test = np.array(x_test)
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape(len(x_test), np.prod(x_test.shape[1:]))
y_train = [map_classes[_y] for _y in y_train]
y_test = [map_classes[_y] for _y in y_test]
y_train = np_utils.to_categorical(y_train, 5)
y_test = np_utils.to_categorical(y_test, 5)

model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=50*50))
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

plt.plot(historico.history['accuracy'])
plt.plot(historico.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(historico.history['loss'])
plt.plot(historico.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
