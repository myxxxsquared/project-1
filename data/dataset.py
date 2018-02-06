import numpy as np
import tensorflow as tf
from .utils import get_maps
from .utils import resize

PKL_DIR = '/home/rjq/data_cleaned/pkl/'
import pickle

def get_train_input():
    dataset = tf.data.Dataset.range(1000)
    dataset = dataset.map(
        lambda index: tuple(tf.py_func(
            wrapper, [index], [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,])))
    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()
    return features

def wrapper(index):
    res = pickle.load(open(PKL_DIR + 'totaltext_train/' + str(index) + '.bin', 'rb'))
    img_name = res['img_name']
    img = res['img']
    cnts = res['contour']
    is_text_cnts = res['is_text_cnts']
    img, cnts = resize(img, cnts, 512, 512)
    img_name, img, maps = data_labeling(img_name,img, cnts, is_text_cnts, 0.2,1.0,5)
    [TR, TCL, radius, cos_theta, sin_theta] = maps
    img = np.array(img, np.float32)
    TR = np.array(TR, np.float32)
    TCL = np.array(TCL, np.float32)
    radius = np.array(radius, np.float32)
    cos_theta = np.array(cos_theta, np.float32)
    sin_theta = np.array(sin_theta, np.float32)
    return img, TR, TCL, radius, cos_theta, sin_theta

def data_labeling(img_name, img, cnts, is_text_cnts, thickness, crop_skel, neighbor, chars = None):
    '''
    :param img_name: pass to return directly, (to be determined, int or str)
    :param img: ndarray, np.uint8,
    :param cnts:
            if is_text_cnts is True: list(ndarray), ndarray: dtype np.float32, shape [n, 1, 2], order(col, row)
            if is_text_cnts is False: list(list(ndarray), list(ndarray)), for [char_cnts, text_cnts]
    :param is_text_cnts: bool
    :param left_top: for cropping
    :param right_bottom: for cropping
    :param chars:
            if is_text_cnts is True: None
            if is_text_cnts is False: a nested list storing the chars info for synthtext
    :return:
            img_name: passed down
            img: np.ndarray np.uint8
            maps: [TR, TCL, radius, cos_theta, sin_theta], all of them are 2-d array,
            TR: np.bool; TCL: np.bool; radius: np.float32; cos_theta/sin_theta: np.float32
    '''

    skels_points, radius_dict, score_dict, cos_theta_dict, sin_theta_dict, mask_fills = \
        get_maps(img, cnts, is_text_cnts, thickness, crop_skel, neighbor, chars)
    TR = mask_fills[0]
    for i in range(1, len(mask_fills)):
        TR = np.bitwise_or(TR, mask_fills[i])
    TCL = np.zeros(img.shape[:2], np.bool)
    for point, _ in score_dict.items():
        TCL[point[0], point[1]] = True
    radius = np.zeros(img.shape[:2], np.float32)
    for point, r in radius_dict.items():
        radius[point[0], point[1]] = r
    cos_theta = np.zeros(img.shape[:2], np.float32)
    for point, c_t in cos_theta_dict.items():
        cos_theta[point[0], point[1]] = c_t
    sin_theta = np.zeros(img.shape[:2], np.float32)
    for point, s_t in sin_theta_dict.items():
        sin_theta[point[0], point[1]] = s_t

    maps = [TR, TCL, radius, cos_theta, sin_theta]
    return img_name, img, maps