import tensorflow as tf
import pickle
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
                                    None)


def wrapper(index):
    file = PKL_DIR + 'totaltext_train/' + str(index) + '.bin'
    img_name, img, maps = data_label(data_aug(load_file(file), augment_rate=100))
    [TR, TCL, radius, cos_theta, sin_theta] = maps
    return img, TR, TCL, radius, cos_theta, sin_theta


def get_train_input():
    dataset = tf.data.Dataset.range(1000)
    dataset = dataset.map(
        lambda index: tuple(tf.py_func(
            wrapper, [index], [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,])))
    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()
    return features



