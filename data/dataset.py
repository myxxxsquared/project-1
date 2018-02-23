import tensorflow as tf
import numpy as np


def get_train_input(params):

    with tf.device('/cpu:0'):
        features = {'input_img': tf.convert_to_tensor(np.ones((9, 512,512, 3)).astype(np.float32)),
                    'Labels': tf.convert_to_tensor(np.ones((9, 512,512,5)).astype(np.float32))}
    return features



