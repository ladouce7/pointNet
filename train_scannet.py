import os
import sys
import datetime

sys.path.insert(0, './')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras

from models.sem_seg_model import SEM_SEG_Model

tf.random.set_seed(42)


def load_dataset(in_file, batch_size, shuffle):

    assert os.path.isfile(in_file), '[error] dataset path not found'

    n_points = 512     #CHANGE WHEN INPUTTING NEW DATA
    shuffle_buffer = 1000

    def _extract_fn(data_record):

        in_features = {
        'points': tf.io.FixedLenFeature([n_points * 3], tf.float32),
        'labels': tf.io.FixedLenFeature([n_points], tf.int64)
        }

        return tf.io.parse_single_example(data_record, in_features)

    def _preprocess_fn(sample):

        points = sample['points']
        labels = sample['labels']

        points = tf.reshape(points, (n_points, 3))
        labels = tf.reshape(labels, (n_points, 1))

        shuffle_idx = tf.range(points.shape[0])
        shuffle_idx = tf.random.shuffle(shuffle_idx)
        points = tf.gather(points, shuffle_idx)
        labels = tf.gather(labels, shuffle_idx)

        return points, labels

    dataset = tf.data.TFRecordDataset(in_file)
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(_extract_fn)
    dataset = dataset.map(_preprocess_fn)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset


def train():

    model = SEM_SEG_Model(config['batch_size'], config['num_classes'], config['bn'])

    train_ds = load_dataset(config['train_ds'], config['batch_size'], config['shuffle'])
    val_ds = load_dataset(config['val_ds'], config['batch_size'], config['shuffle'])
    
    callbacks = [
    keras.callbacks.TensorBoard(
    './logs/{}'.format(config['log_dir']), update_freq=50),
    keras.callbacks.ModelCheckpoint(
    './logs/{}/model/weights'.format(config['log_dir']), 'val_sparse_categorical_accuracy', save_best_only=True)
    ]

    model.build((config['batch_size'], config['input_event_len'], 3))
    print(model.summary())

    model.compile(
    optimizer=keras.optimizers.Adam(config['lr']),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )

    model.fit(
    train_ds,
    validation_data=val_ds,
    validation_steps=10,
    validation_freq=1,
    callbacks=callbacks,
    epochs=100,
    verbose=1
    )


if __name__ == '__main__':

    config = {
    'train_ds' : 'data/Mg22_size512_train.tfrecord',    #change
    'val_ds' : 'data/Mg22_size512_val.tfrecord',    #change
    'log_dir' : 'scannet_1',
    'log_freq' : 10,
    'test_freq' : 100,
    'batch_size' : 4,
    'num_classes' : 7,    #change
    'lr' : 0.001,
    'bn' : False,
        'shuffle' : True,
        'input_event_len' : 512    #change
    }

    train()
