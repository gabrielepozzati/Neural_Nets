#!/usr/bin/env python3
import random
import pickle
import keras
import math
import numpy as np
from keras.layers import *
from keras.models import load_model, Model
from keras.engine.topology import Input, Layer

thresholds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999]

def get_accuracy(cm):
    return float(cm['TP']+cm['TN'])/(cm['TP']+cm['TN']+cm['FP']+cm['FN'])

def get_mcc(cm):
    num = float((cm['TP']*cm['TN'])-(cm['FP']*cm['FN']))
    den = math.sqrt(float((cm['TP']+cm['FP'])*(cm['TP']+cm['FN'])*(cm['TN']+cm['FP'])*(cm['TN']+cm['FN'])))
    return num/den

##### Test variables #####
test_path = 'test' 
model_path = 'logs/model_path19'
window_size = 8
encoding_size = 20
input_length = encoding_size*((window_size*2)+1)


model = load_model(model_path)

with open('dataset.pickle','rb') as f:
    dataset = pickle.load(f)

target_list = []
for code in open(test_path, 'r'):
    if code.strip() not in dataset:
        print (code.strip())
        continue
    for n in range(len(dataset[code.strip()])):
        target_list.append(code.strip()+'_'+str(n))

confusion_matrix = {'TP':0, 'FP':0,'TN':0, 'FN':0}
confusion_matrix_dict = {}
for thr in thresholds: confusion_matrix_dict[thr] = confusion_matrix.copy()

for example_code in target_list:
    pdb_code = example_code.split('_')[0]
    example_index = int(example_code.split('_')[1])
    X = np.array(dataset[pdb_code][example_index][0], dtype=np.float64)
    X = X.reshape(1, input_length)
    true_label = int(dataset[pdb_code][example_index][1])
    prediction = model.predict_on_batch(X)
    prediction = prediction[0][0]

    for thr in thresholds:
        if prediction >= thr:
            if true_label == 1: confusion_matrix_dict[thr]['TP'] += 1
            else: confusion_matrix_dict[thr]['FP'] += 1
        if prediction < thr:
            if true_label == 0: confusion_matrix_dict[thr]['TN'] += 1
            else: confusion_matrix_dict[thr]['FN'] += 1

print ('Accuracy: ', get_accuracy(confusion_matrix_dict[0.5]))

