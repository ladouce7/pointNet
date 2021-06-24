'''
This converter is used to convert hdf5 file to TFRecords.

Dataset used:
This is designed for the point cloud hdf5 data of pointnet,
which can be downloaded from https://github.com/charlesq34/pointnet/sem_seg

The sample data shape is 
    h5py {
        'data': (1000, 4096, 9), # (number_of_data, points, channels)
        'label': (1000, 4096)    # (number_of_data, label_of_points)
    }

If you want modify this for your own hdf5 data, 
the only thing you need to modify is "get_feature(point_cloud, label)" function

https://gist.github.com/shijianjian/93b52ae96137fcd3d63edd92eb3f5046
'''

import h5py
import tensorflow as tf
import numpy as np


# For array storage, TFRecords will only support list storage or 1-D array storage
# If you have multi-dimensional array, please start with:
#     array = array.reshape(-1)
def _int64_feature(value):
    #value = list(map(int, value))
    #value = np.array(list(map(int, value)))
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _floats_feature(value):
    value = np.reshape(value, -1)
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def get_all_keys_from_h5(h5_file):
    res = []
    for key in h5_file.keys():
        res.append(key)
    return res

# details of 9-D vector: https://github.com/charlesq34/pointnet/issues/7
def get_feature(point_cloud):
    return {
        #'x': _floats_feature(point_cloud[:, 0]),
        #'y': _floats_feature(point_cloud[:, 1]),
        #'z': _floats_feature(point_cloud[:, 2]),
        'points': _floats_feature(point_cloud[:, :3]),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(point_cloud[0,5])]))
    }

def h5_to_tfrecord_converter(input_file_path, output_file_path):
    h5_file = h5py.File(input_file_path, 'r')
    keys = get_all_keys_from_h5(h5_file)
    
    num_of_instances = h5_file[keys[0]][:].shape[0]   #in each event

    # Check the number of values in each event
    for key in keys:
        if h5_file[key][:].shape[0] != num_of_instances:
            raise ValueError('Invalid values. The inequality of the number of values in each key.')

    with tf.io.TFRecordWriter(output_file_path) as writer:
        for index in range(len(keys)):
            event = keys[index]
            example = tf.train.Example(
                features=tf.train.Features(
                    feature = get_feature(h5_file[event])
              ))
            writer.write(example.SerializeToString())
            print('\r{:.1%}'.format((index+1)/num_of_instances), end='')
    
# With commandline enabled
if __name__ == "__main__":

    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file-path', required=True, help='Path to the input HDF5 file.')
    parser.add_argument('--output-file-path', default='', help='Path to the output TFRecords.')
    parser.add_argument('-r', action='store_true', help='Recursively find *.h5 files under pointed folder. This will not dive deeper into sub-folders.')
    FLAGS = parser.parse_args()

    INPUT_PATH = os.path.abspath(FLAGS.input_file_path)
    OUTPUT_PATH = FLAGS.output_file_path
    RECURSIVE = FLAGS.r

    if not INPUT_PATH.endswith('.h5') and not RECURSIVE:
        raise ValueError('Not a valid HDF5 file provided, you may want to add -r.')

    elif INPUT_PATH.endswith('.h5'):
        if OUTPUT_PATH == '':
            OUTPUT_PATH = INPUT_PATH[:-3]
        print('Start converting...\t')
        h5_to_tfrecord_converter(INPUT_PATH, os.path.abspath(OUTPUT_PATH) + '.tfrecord')

    elif RECURSIVE:
        files = []
        if OUTPUT_PATH == '':
            OUTPUT_PATH = INPUT_PATH
        for _file in os.listdir(INPUT_PATH):
            if _file.endswith('.h5'):
                files.append((
                    os.path.join(INPUT_PATH, _file[:-3]), 
                    os.path.join(os.path.abspath(OUTPUT_PATH), _file[:-3]),
                    _file
                    ))
        print(len(files), 'of HDF5 file detected.')
        for idx, (_input, _output, _file_name) in enumerate(files):
            print('\n\ton job %d/%d, %s' % (idx, len(files), _file_name), end='')
            h5_to_tfrecord_converter(_input + '.h5', _output + '.tfrecord')

    else:
        pass