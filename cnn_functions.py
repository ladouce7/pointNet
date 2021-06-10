import os
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
import numpy as np
import annabel as anna
import cnn_functions as cnn
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
layers = tf.keras.layers


#builds a VGG16 CNN model with some extra layers
def build_pretrained_vgg_model(input_shape, num_classes):
    vgg_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    net = layers.Flatten()(vgg_model.output)
    net = layers.Dense(512, activation=tf.nn.relu)(net)   #two fully-connected layers that take in the flattened features
    net = layers.Dense(512, activation=tf.nn.relu)(net)
    output = layers.Dense(num_classes, activation=tf.nn.softmax)(net)   #groups images into one of three classes
    model = tf.keras.Model(inputs=vgg_model.input, outputs=output)   #model whose input is VGG16's and output is that of the layer above
    for layer in model.layers[:-4]:    #specifying that the last four layers (the ones we've made) won't be updated during training
        layer.trainable = False  
    return model

#plots a learning curve from a training history
def plot_learning_curve(history):
    plt.figure(figsize=(11, 6), dpi=100)
    plt.plot(history.history['loss'], 'o-', label='Training Loss')
    plt.plot(history.history['val_loss'], 'o:', color='r', label='Validation Loss')
    plt.legend(loc='best')
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(range(0, len(history.history['loss'])), range(1, len(history.history['loss']) + 1))
    plt.show()