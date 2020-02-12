#!/usr/bin/env python3
import pickle
import random
import numpy as np
import keras
from keras.layers import *
from keras import backend as K
from keras.optimizers import Adam
from keras.models import load_model, Model
from keras.engine.topology import Input, Layer

def model_set(input_length, learning_rate):

    input_layer = Input(shape=(input_length,))

    hidden_layer = Dense(64, activation='relu')(input_layer)

    output_layer = Dense(1, activation='sigmoid')(hidden_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    adam = Adam(lr=learning_rate)

    model.compile(loss='binary_crossentropy', optimizer = adam)

    return model

thresholds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999]

def get_accuracy(cm):
    return float(cm['TP']+cm['TN'])/(cm['TP']+cm['TN']+cm['FP']+cm['FN'])

##### Training data variables #####
training_list_file = 'train'
validation_list_file = 'validation'
window_size = 8
encoding_size = 20

##### Training procedure variables #####
epoch_number = 20
mini_batch_size = 512
learning_rate = 0.005
input_length = encoding_size*((window_size*2)+1)

##### Neural Network architecture #####
model = model_set(input_length, learning_rate)

##### Load Dataset #####
with open('dataset.pickle','rb') as f:
    dataset = pickle.load(f)

training_list = []
for code in open(training_list_file, 'r'):
    if code.strip() not in dataset: 
        print (code.strip()+'')
        continue
    for n in range(len(dataset[code.strip()])):
        training_list.append(code.strip()+'_'+str(n))

validation_list = []
for code in open(validation_list_file, 'r'):
    if code.strip() not in dataset:
        print (code.strip())
        continue
    for n in range(len(dataset[code.strip()])):
        validation_list.append(code.strip()+'_'+str(n))

##### Training - Epoch cycling setup ####
best = 0
mini_batch = []
for n in range(1, epoch_number+1):
    random.shuffle(training_list)
    print('\nEpoch '+str(n)+'/'+str(epoch_number)+' --- Learning Rate >>> '+str(K.get_value(model.optimizer.lr)))
    epoch_losses = []

    ##### Training - Batch formatting #####
    for example_code in training_list:
        pdb_code = example_code.split('_')[0]
        example_index = int(example_code.split('_')[1])
        mini_batch.append(dataset[pdb_code][example_index])
        if len(mini_batch) > mini_batch_size-1:
            mbX = []
            mbY = []
            for example in mini_batch:
                mbX.append(example[0])
                mbY.append(example[1])
            X = np.array(mbX, dtype=np.float64)
            Y = np.array(mbY, dtype=np.float64)
            X = X.reshape(mini_batch_size, input_length)
            Y = Y.reshape(mini_batch_size, 1)

            ##### Training - Model weights fit #####
            loss = model.train_on_batch(X, Y)
            epoch_losses.append(loss)
            mini_batch = []

    ##### Prediction/evaluation of validation set #####
    print ('Epoch '+str(n)+' complete! Evaluation...')
    confusion_matrix = {'TP':0, 'FP':0,'TN':0, 'FN':0}
    confusion_matrix_dict = {}
    for thr in thresholds: confusion_matrix_dict[thr] = confusion_matrix

    for example_code in validation_list:
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

    score = get_accuracy(confusion_matrix_dict[0.5])

    ##### Model save, stats display #####
    acc = 0
    for el in epoch_losses: acc += el

    outfile = open('logs/history','a')
    if score >= best:
        model.save('logs/model_epoch'+str(n))
        print ('Loss: '+str(acc/len(epoch_losses))+' -  Val score: '+str(score)+' >>> Partial saved!')
        outfile.write('Epoch_'+str(n)+' Loss: '+str(acc/len(epoch_losses))+' - Val score: '+str(score)+' >>> Partial saved!\n')
        best = score
    else:
        print ('Loss: '+str(acc/len(epoch_losses))+' -  Val score: '+str(score))
        outfile.write('Epoch_'+str(n)+' Loss: '+str(acc/len(epoch_losses))+' - Val score: '+str(score)+'\n')
    outfile.close()

    #K.set_value(model.optimizer.lr, LR*(0.99-(0.01*n)))
