import json
import cv2
import os
import time
import numpy as np

from PIL import Image, ImageDraw

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score

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

def word_value(word):
  if   word == 'cloud': return 1
  elif word == 'canoe': return 2
  elif word == 'bread': return 3
  elif word == 'bench': return 4
  return 5
  
def fit(data, model, data_test, label_test):
  
  isFristFit = len(data_test) == 0
  
  x = [get_blob(_data['drawing']).flatten() for _data in data]
  y = np.full(len(x), word_value(data[0]['word']))
  
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

  data_test.extend(x_test)
  label_test.extend(y_test)

  if isFristFit:
    model.partial_fit(x_train, y_train, np.array([1, 2, 3, 4, 5]))
  else: 
    model.partial_fit(x_train, y_train)

  del x
  del y
  del x_train
  del y_test

def fit_in_batch(data_list, batch_size, model, data_test, label_test):
  
  last_index = 0
  count_batch = 1
  list_size = len(data_list)
  
  while (last_index < list_size):
    
    max_index = last_index + batch_size

    if max_index > list_size: max_index = list_size
    #print('run batch train number ', count_batch, ' total trained items ', max_index)

    fit(data_list[last_index:max_index], model, data_test, label_test)
    
    last_index = max_index
    count_batch+=1

def fit_image(ndjson, model, data_test, label_test):
  with open(ndjson) as file:
    
    jsons = file.read().strip().split('\n')
    jsons = [json.loads(j) for j in jsons]
    
    file.close()
    batch_size = 100

    fit_in_batch(jsons, batch_size, model, data_test, label_test)
    del jsons

def print_metrics(model, data_test, label_test, train_time):

  start_predict_time = time.time()
  predict            = model.predict(data_test)
  predict_time       = time.time() - start_predict_time

  accuracy  = accuracy_score(label_test, predict)
  f1        = f1_score(label_test, predict, average='macro')
  precision = precision_score(label_test, predict, average='macro')

  print("Accuracy: %.2f") % (accuracy)
  print("F1: %.2f") % (f1)
  print("Precision: %.2f") % (precision)
  print("Train time %.2f seconds") % (train_time)
  print("Predict time %.2f seconds") % (predict_time)

def main():

  var_smoothing_list = [1e-9, 1e-8, 1e-0]
  data_set_folder = "data_set"
  
  for var_smoothing in var_smoothing_list:

    print('----------------------------------------------------------------------')
    print('train init for var_smoothing ', var_smoothing)

    data_test = []
    label_test = []
    model = GaussianNB(var_smoothing=var_smoothing)
    star_train_time = time.time()

    for data_set_file in os.listdir("data_set"):
      data_set_path = data_set_folder+"/"+data_set_file
      print('----------------------------------------------------------------------')
      print('fit ', data_set_path)
      fit_image(data_set_path, model, data_test, label_test)

    print('----------------------------------------------------------------------')
    print('train finished')
    train_time = time.time() - star_train_time
    print_metrics(model, data_test, label_test, train_time)

main()
