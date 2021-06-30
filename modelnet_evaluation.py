import cnn_functions as cnn
import numpy as np
from models.cls_msg_model import CLS_MSG_Model
from train_modelnet import load_dataset
import h5py

data_name = 'Mg22_size128'

config = {
    'train_ds' : 'data/' + data_name + '_train.tfrecord',   
    'val_ds' : 'data/' + data_name + '_val.tfrecord',   
    'log_dir' : 'msg_1',
    'batch_size' : 8,
    'lr' : 0.00001,
    'num_classes' : 6,
    'msg' : True,
    'bn' : False,
    'shuffle' : False,
    'input_event_len' : 128     #change this for new data
    }

if config['msg'] == True:
        model = CLS_MSG_Model(config['batch_size'], config['num_classes'], config['bn'])
else:
        model = CLS_SSG_Model(config['batch_size'], config['num_classes'], config['bn'])

model.build(input_shape=(config['batch_size'], config['input_event_len'], 3))
#print(model.summary())

model.load_weights('./logs/msg_1/model/weights.ckpt')

test_ds_h5 = h5py.File(data_name + '_test.h5', 'r')    #change
keys = list(test_ds_h5.keys())
length = len(keys)
real = np.zeros(length,int)
for i in range(length):
    key = keys[i]
    event = test_ds_h5[key]
    real[i] = event[0,-1]
    
test_ds = load_dataset('data/' + data_name + '_test.tfrecord', config['batch_size'], config['shuffle'])    #change this for new data
predicted_probabilities = model.predict(test_ds, batch_size=config['batch_size'])
predictions = np.argmax(predicted_probabilities, axis=1)
class_names = ['beam', 'two track', 'three track', 'four track', 'five track', 'six track']

if real.shape > predictions.shape:
    diff = real.shape - predictions.shape
    real = real[-diff]
    
np.save('RealLabels', real)
np.save('PredictedLabels', predictions)

cnn.plot_confusion_matrix(real, predictions, class_names)