import cnn_functions as cnn
import numpy as np
from models.sem_seg_model import SEM_SEG_Model
from train_scannet import load_dataset
import h5py

sample_size = 1024
    
config = {
    'train_ds' : 'data/Mg22_size' + str(sample_size) + '_train.tfrecord',   
    'val_ds' : 'data/Mg22_size' + str(sample_size) + '_val.tfrecord',   
    'log_dir' : 'scannet_1',
    'log_freq' : 10,
    'test_freq' : 100,
    'batch_size' : 16,
    'num_classes' : 7,    #CHANGE
    'lr' : 0.001,
    'bn' : False,
    'shuffle' : False,
    'input_event_len' : sample_size    
    }

model = SEM_SEG_Model(config['batch_size'], config['num_classes'], config['bn'])

model.build(input_shape=(config['batch_size'], config['input_event_len'], 3))
#print(model.summary())

model.load_weights('./logs/scannet_1/model/weights')

test_ds_h5 = h5py.File('Mg22_size' + str(sample_size) + '_test.h5', 'r')    #change
keys = list(test_ds_h5.keys())
length = len(keys)
real = np.zeros((length, sample_size),int)
for i in range(length):
    key = keys[i]
    event = test_ds_h5[key]
    real[i] = event[:,5]
#make sure labels aren't one hot encoded
#concatenate the predictions vector for each 
    
test_ds = load_dataset('data/Mg22_size' + str(sample_size) + '_test.tfrecord', config['batch_size'], config['shuffle'])
predicted_probabilities = model.predict(test_ds, batch_size=config['batch_size'])
predictions = np.argmax(predicted_probabilities, axis=1)

class_names = ['beam', '1', '2', '3', '4', '5', '6']

if real.shape[0] > predictions.shape[0]:
    diff = real.shape[0] - predictions.shape[0]
    real = real[:-diff,:]
    
np.save('RealLabels_Mg22_size' + str(sample_size), real)
np.save('PredictedLabels_Mg22_size' + str(sample_size), predictions)

cnn.plot_confusion_matrix(real, predictions, class_names)