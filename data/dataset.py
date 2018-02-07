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
                                    ins.get('chars',None))


def syn_wrapper(index):
    file = PKL_DIR + 'synthtext_chars/' + str(index) + '.bin'
    img_name, img, maps = data_label(data_aug(load_file(file), augment_rate=100))
    [TR, TCL, radius, cos_theta, sin_theta] = maps
    img = np.array(img, np.float32)
    TR = np.array(TR, np.float32)
    TCL = np.array(TCL, np.float32)
    radius = np.array(radius, np.float32)
    cos_theta = np.array(cos_theta, np.float32)
    sin_theta = np.array(sin_theta, np.float32)

    # res = np.stack((img, TR, TCL, radius, cos_theta, sin_theta))
    return img, TR, TCL, radius, cos_theta, sin_theta


def get_train_input(params):
    syn_dataset = tf.data.Dataset.range(858749+1).repeat(params.pretrain_num)

    syn_dataset = syn_dataset.map(
        lambda index: tuple(tf.py_func(
            syn_wrapper, [index], [tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32])))

    # total_train_dataset = tf.data.Dataset.range(858749+1).repeat(params.pretrain_num)
    # total_train_dataset = total_train_dataset.map(
    #     lambda index: tuple(tf.py_func(
    #         total_train_wrapper, [index], [tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32])))
    #
    # total_test_dataset = tf.data.Dataset.range(858749+1).repeat(params.pretrain_num)
    # total_test_dataset = total_test_dataset.map(
    #     lambda index: tuple(tf.py_func(
    #         total_test_wrapper, [index], [tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32])))


    # syn_dataset = syn_dataset.batch(32)
    iterator = syn_dataset.make_one_shot_iterator()

    features = iterator.get_next()
    return features



