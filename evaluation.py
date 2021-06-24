import cnn_functions as cnn
import numpy as np
from models.cls_msg_model import CLS_MSG_Model
from train_modelnet import load_dataset
import h5py

config = {
    'train_ds' : 'data/AllEvents_size50_train.tfrecord',
    'val_ds' : 'data/AllEvents_size50_val.tfrecord',
    'log_dir' : 'msg_1',
    'batch_size' : 4,
    'lr' : 0.001,
    'num_classes' : 3,
    'msg' : True,
    'bn' : False
    }

if config['msg'] == True:
        model = CLS_MSG_Model(config['batch_size'], config['num_classes'], config['bn'])
else:
        model = CLS_SSG_Model(config['batch_size'], config['num_classes'], config['bn'])

model.build(input_shape=(config['batch_size'], 50, 3))
#print(model.summary())

model.load_weights('./logs/msg_1/model/weights.ckpt')

test_ds_h5 = h5py.File('AllEvents_size50_test.h5', 'r')
keys = list(test_ds_h5.keys())
length = len(keys)
real = np.zeros(length,int)
for i in range(length):
    key = keys[i]
    event = test_ds_h5[key]
    real[i] = event[0,5]
real = real[:-2]
    
test_ds = load_dataset('data/AllEvents_size50_test.tfrecord', config['batch_size'])
predicted_probabilities = model.predict(test_ds, batch_size=config['batch_size'])
predictions = np.argmax(predicted_probabilities, axis=1)
class_names = ['beam', 'two track', 'three track']
print(real.shape, predictions.shape)
if real.shape > predictions.shape:
    diff = real.shape - predictions.shape
    real = real[-diff]

cnn.plot_confusion_matrix(real, predictions, class_names)