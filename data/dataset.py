import tensorflow as tf
import pickle
import numpy as np
from .data_augementation import DataAugmentor
from .data_labeling import data_churn

PKL_DIR = '/home/rjq/data_cleaned/pkl/'

DA = DataAugmentor()
labelling = data_churn()


def load_file(file):
    return pickle.load(open(file,'rb'))


def data_aug(ins,augment_rate):
    return DA.augment(ins,augment_rate=augment_rate)


def data_label(ins):
    return labelling._data_labeling(ins['img_name'],ins['img'],
                                    ins['contour'],ins['is_text_cnts'],
                                    ins['left_top'],ins['right_bottom'],
                                    ins['chars'])


def wrapper(index):
    file = PKL_DIR + 'totaltext_train/' + str(index) + '.bin'
    img_name, img, maps = data_label(data_aug(load_file(file), augment_rate=100))
    [TR, TCL, radius, cos_theta, sin_theta] = maps
    # img = tf.cast(img, tf.float32)
    # TR = tf.cast(TR, tf.float32)
    # TCL = tf.cast(TCL, tf.float32)
    radius = np.array(radius, np.float32)
    # cos_theta = tf.cast(cos_theta, tf.float32)
    # sin_theta = tf.cast(sin_theta, tf.float32)
    return radius


def get_train_input():
    dataset = tf.data.Dataset.range(1000)
    dataset = dataset.map(
        lambda index: tuple(tf.py_func(
            wrapper, [index], [tf.float32])))
    iterator = dataset.make_one_shot_iterator()
    dataset = dataset.batch(32)
    features = iterator.get_next()
    return features



