import os
import errno
import numpy as np
import pickle
import random

import tensorflow as tf

def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

class InputData(object):
    def __init__(self, train_images_path, valid_images_path, anomaly_images_path, test_images_path):

        self.dataname = "InputData"
        self.image_size = 0
        self.channel = 0
        self.train_images_path = train_images_path
        self.valid_images_path = valid_images_path
        self.test_images_path = test_images_path        
        self.train_data_list = self.load_input_pickle(train_images_path)
        self.valid_data_list = self.load_input_pickle(valid_images_path)
        self.anomaly_data_list = self.load_input_pickle(anomaly_images_path)
        self.test_data_list = self.load_input_pickle(test_images_path)

    def load_input_pickle(self, pickle_path):
            
        if pickle_path==None:
            return None

        features, labels = pickle.load(open(pickle_path, mode='rb'))

        if len(features.shape) == 3:
            features = np.expand_dims(features, axis=-1)

        return (features, labels)

def get_batch(data, batch_size):
    
    idx = np.array(list(range(0, len(data[0]))))
    
    random.shuffle(idx)
    next_x = data[0][idx[0:batch_size]]
    next_y = data[1][idx[0:batch_size]]     
    
    return next_x, next_y
    
