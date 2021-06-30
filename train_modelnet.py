import os
import sys
import datetime
import cnn_functions as cnn

sys.path.insert(0, './')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras

from models.cls_msg_model import CLS_MSG_Model
from models.cls_ssg_model import CLS_SSG_Model

tf.random.set_seed(1234)


def load_dataset(in_file, batch_size, shuffle):

    assert os.path.isfile(in_file), '[error] dataset path not found'

    n_points = 512     #change when inputting new data
    shuffle_buffer = 1000
    
    def _extract_fn(data_record):
        in_features = {
            'points': tf.io.FixedLenFeature([n_points*3], tf.float32),
            'label': tf.io.FixedLenFeature([1], tf.int64)
        }
        return tf.io.parse_single_example(data_record, in_features)

    def _preprocess_fn(sample):
        points = sample['points']
        label = sample['label']
        points = tf.reshape(points, (n_points, 3))
        points = tf.random.shuffle(points)

        return points, label

    dataset = tf.data.TFRecordDataset(in_file)
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(_extract_fn)
    dataset = dataset.map(_preprocess_fn)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset


def train():

    if config['msg'] == True:
        model = CLS_MSG_Model(config['batch_size'], config['num_classes'], config['bn'])
    else:
        model = CLS_SSG_Model(config['batch_size'], config['num_classes'], config['bn'])

    train_ds = load_dataset(config['train_ds'], config['batch_size'], config['shuffle'])
    val_ds = load_dataset(config['val_ds'], config['batch_size'], config['shuffle'])

    callbacks = [
    keras.callbacks.EarlyStopping(
    'val_sparse_categorical_accuracy', min_delta=0.01, patience=10),
    keras.callbacks.TensorBoard(
    './logs/{}'.format(config['log_dir']), update_freq=50),
    keras.callbacks.ModelCheckpoint(
    './logs/{}/model/weights.ckpt'.format(config['log_dir']), 'val_sparse_categorical_accuracy', save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001)
    ]

    model.build(input_shape=(config['batch_size'], config['input_event_len'], 3))     
    print(model.summary())

    model.compile(
    optimizer=keras.optimizers.Adam(config['lr']),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )

    history = model.fit(
    train_ds,
    validation_data = val_ds,
    validation_steps = 20,
    validation_freq = 1,
    callbacks = callbacks,
    epochs = 100,
    verbose = 1
    )
    cnn.plot_learning_curve(history)

if __name__ == '__main__':

    config = {
    'train_ds' : 'data/Mg22_size512_train.tfrecord',     #change
    'val_ds' : 'data/Mg22_size512_val.tfrecord',    #change
    'log_dir' : 'msg_1',
    'batch_size' : 8,
    'lr' : 0.001,
    'num_classes' : 6,    #change
    'msg' : True,
    'bn' : False,
    'shuffle' : True,
    'input_event_len' : 128
    }

    train()
