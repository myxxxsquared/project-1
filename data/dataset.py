import tensorflow as tf
import pickle
import numpy as np
from .data_augementation import augment
from .utils import data_labeling


def load_file(file):
    return pickle.load(open(file,'rb'))


def data_aug(ins,augment_rate):
    return augment(ins, augment_rate=augment_rate)


def data_label(ins):
    return data_labeling(ins['img_name'],ins['img'],
                                    ins['contour'],ins['is_text_cnts'],
                                    ins['left_top'],ins['right_bottom'],
                                    ins['chars'])


def syn_wrapper(index):
    PKL_DIR = '/home/rjq/data_cleaned/pkl/'
    file = PKL_DIR + 'synthtext_chars/' + str(index) + '.bin'
    img_name, img, maps = data_label(data_aug(load_file(file), augment_rate=100))
    TR, TCL, radius, cos_theta, sin_theta = maps

    # import time
    # time.sleep(5)
    # img = np.zeros((512,512,3))
    # TR = np.zeros((512,512,1))
    # TCL = np.zeros((512,512,1))
    # radius = np.zeros((512,512,1))
    # cos_theta = np.zeros((512,512,1))
    # sin_theta = np.zeros((512,512,1))
    # img = np.reshape(np.array(img, np.float32),(512,512,3))
    # TR = np.reshape(np.array(TR, np.float32),(512,512,1))
    # TCL = np.reshape(np.array(TCL, np.float32),(512,512,1))
    # radius = np.reshape(np.array(radius, np.float32),(512,512,1))
    # cos_theta = np.reshape(np.array(cos_theta, np.float32),(512,512,1))
    # sin_theta = np.reshape(np.array(sin_theta, np.float32),(512,512,1))

    img = np.reshape(np.array(img, np.float32),(512,512,3))
    TR = np.reshape(np.array(TR, np.float32),(512,512,1))
    TCL = np.reshape(np.array(TCL, np.float32),(512,512,1))
    radius = np.reshape(np.array(radius, np.float32),(512,512,1))
    cos_theta = np.reshape(np.array(cos_theta, np.float32),(512,512,1))
    sin_theta = np.reshape(np.array(sin_theta, np.float32),(512,512,1))

    # print(img.shape)
    # res = np.stack((img, TR, TCL, radius, cos_theta, sin_theta))
    return img, TR, TCL, radius, cos_theta, sin_theta


def get_train_input(params):
    features = {'input_img': tf.convert_to_tensor(np.ones((9, 512,512, 3)).astype(np.float32)),
                'Labels': tf.convert_to_tensor(np.ones((9, 512,512,5)).astype(np.float32))}
    return features



