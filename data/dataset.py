import tensorflow as tf
import os
import numpy as np
from data.data_augmentation import DataAugmentor
from data.data_labelling import data_churn
import multiprocessing as mp
import pickle
import gzip

TOTAL_TRAIN_DIR = '/home/rjq/data_cleaned/pkl/totaltext_train_care/'
TOTAL_TEST_DIR = '/home/rjq/data_cleaned/pkl/totaltext_test_care/'


DA = DataAugmentor()
labelling = data_churn()


def _load_file(file, is_syn):
    if not is_syn:
        if file.endswith('gz'):
            return pickle.load(gzip.open(file, 'rb'))
        else:
            return pickle.load(open(file, 'rb'))
    else:
        if file.endswith('gz'):
            return pickle.load(gzip.open(file, 'rb'), encoding='latin1')
        else:
            return pickle.load(open(file, 'rb'), encoding='latin1')


def _data_aug(ins, augment_rate, test_mode=False, real_test=False):
    return DA.augment(ins, augment_rate=augment_rate, test_mode=test_mode, real_test=real_test)


def _data_label(ins, is_pixellink):
    if is_pixellink:
        data = labelling._pixellink_labeling(ins['img_name'], ins['img'],
                                        ins['contour'], ins['is_text_cnts'],
                                        ins['left_top'], ins['right_bottom'],
                                        ins.get('chars', None), ins['care'])
    else:
        data = labelling._data_labeling(ins['img_name'], ins['img'],
                                        ins['contour'], ins['is_text_cnts'],
                                        ins['left_top'], ins['right_bottom'],
                                        ins.get('chars', None), ins['care'])
    return data


def loading_data(file, test_mode=False, real_test=False, is_syn=False, is_pixellink=True):
    return _data_label(_data_aug(_load_file(
        file, is_syn=is_syn),
        augment_rate=1, test_mode=test_mode, real_test=real_test),
        is_pixellink)


def _decompress(ins):
    name = ins[0]
    img = ins[1]
    non_zero, radius, cos, sin = ins[2]
    maps = np.zeros((*(img.shape[:2]), 5), np.float32)
    maps[:, :, 4] = np.cast['uint8'](ins[3])  # -->TR
    maps[:, :, 0][non_zero] = 1               # -->TCL
    maps[:, :, 1][non_zero] = radius          # -->radius
    maps[:, :, 2][non_zero] = cos             # -->cos
    maps[:, :, 3][non_zero] = sin             # -->TCL
    cnt = ins[4]
    return (name, img, maps, cnt)


def load_pre_gen(file):
    return _decompress(pickle.load(gzip.open(file, 'rb')))



####on line data###########
Q = mp.Queue(maxsize=3000)
print('queue excuted')

def enqueue(file_name, test_mode=False, real_test=False, is_syn=False, is_pixellink=True):
    img_name, img, cnts, maps = loading_data(file_name, test_mode, real_test, is_syn, is_pixellink)
    print(img.shape)
    print(maps.shape)
    Q.put({'input_img': img,
           'Labels': maps.astype(np.float32)})


def start_queue(params):
    thread_num = 10# params.thread_num
    file_names_totaltext_train = [TOTAL_TRAIN_DIR+name for name in os.listdir(TOTAL_TRAIN_DIR)]# *params.pre_epoch

    print('start')
    pool = mp.Pool(thread_num)
    pool.map_async(enqueue,file_names_totaltext_train)
    print(Q.qsize())
    print('end')

def get_generator(aqueue):
    def func():
        while True:
            features = aqueue.get()
            yield {'input_img': features['input_img'].astype(np.float32),
                    'Labels': features['Labels'].astype(np.float32)}
    return func


def get_train_input(params):
    g = get_generator(Q)
    train_dataset = tf.data.Dataset.from_generator(g, {'input_img':tf.float32,
                                                        'Labels': tf.float32},
                                                   {'input_img': (tf.Dimension(None),tf.Dimension(None),tf.Dimension(None)),
                                                    'Labels': (tf.Dimension(None),tf.Dimension(None),tf.Dimension(None))}
                                                   )
    train_dataset = train_dataset.batch(params.batch_size).prefetch(params.buffer)
    iterator = train_dataset.make_one_shot_iterator()
    features = iterator.get_next()
    return features


#######solution mp.queue##########
# Q = mp.Queue(maxsize=3000)
# print('queue excuted')
#
# def enqueue(file_name):
#     img_name, img, maps, cnts = load_pre_gen(file_name)
#     Q.put({'input_img': img,
#            'Labels': maps.astype(np.float32)})
#
#
# def start_queue(params):
#     thread_num = params.thread_num
#     file_names_syn = [SYN+name for name in os.listdir(SYN)]*params.pre_epoch
#     file_names_total = [TOTAL_TRAIN+name for name in os.listdir(TOTAL_TRAIN)]*params.epoch
#     file_names = file_names_syn+file_names_total
#
#     print('start')
#     pool = mp.Pool(thread_num)
#     for file_name in file_names:
#         pool.apply_async(enqueue, (file_name,))
#     print('end')
#
#
# def get_generator(params, aqueue):
#     def func():
#         while True:
#             features = aqueue.get()
#             yield {'input_img': features['input_img'].astype(np.float32),
#                     'Labels': features['Labels'].astype(np.float32)}
#     return func
#
#
# def get_train_input(params):
#     g = get_generator(params, Q)
#     train_dataset = tf.data.Dataset.from_generator(g, {'input_img':tf.float32,
#                                                         'Labels': tf.float32},
#                                                    {'input_img': (tf.Dimension(None),tf.Dimension(None),tf.Dimension(None)),
#                                                     'Labels': (tf.Dimension(None),tf.Dimension(None),tf.Dimension(None))}
#                                                    )
#     train_dataset = train_dataset.batch(params.batch_size).prefetch(params.buffer)
#     iterator = train_dataset.make_one_shot_iterator()
#     features = iterator.get_next()
#     return features

########solution map###############
# def wrapper(index, file_names):
#
#     img_name, img, maps, cnts = load_pre_gen(file_names[index])
#     return img, maps.astype(np.float32)
#     # return {'input_img': img,
#     #        'Labels': maps.astype(np.float32)}
#
#
#
# def get_train_input(params):
#     file_names_syn = [SYN+name for name in os.listdir(SYN)]*params.pre_epoch
#     file_names_total = [TOTAL_TRAIN+name for name in os.listdir(TOTAL_TRAIN)]*params.epoch
#     file_names = file_names_syn+file_names_total
#
#     train_dataset = tf.data.Dataset.range(len(file_names))
#
#     train_dataset = train_dataset.map(lambda index: tuple(tf.py_func(
#         wrapper, [index, file_names], (tf.uint8, tf.float32))),
#           num_parallel_calls=params.thread_num).batch(params.batch_size).prefetch(params.buffer)
#
#     # train_dataset = train_dataset.map(lambda index: wrapper(index, file_names))
#     iterator = train_dataset.make_one_shot_iterator()
#     features = iterator.get_next()
#     features = {'input_img':tf.reshape(features[0], [-1]+params.input_size),
#                 'Labels':tf.reshape(features[1], [-1]+params.Label_size)}
#     return features


def _pad_cnts(cnts, cnt_point_max):
    new = []
    for cnt_ in cnts:
        if len(cnt_) < cnt_point_max:
            new.append(np.concatenate((cnt_, np.zeros([cnt_point_max-len(cnt_), 1, 2])), 0))
        else:
            new.append(cnt_)
    return new


def generator_eval():
    file_names = [TOTAL_TEST_DIR+name for name in os.listdir(TOTAL_TEST_DIR)]
    for file_name in file_names:
        img_name, img, maps, cnts = loading_data(file_name, True, False)
        features = {}
        features["input_img"] = np.expand_dims(img,0).astype(np.float32)
        lens = np.array([cnt.shape[0] for cnt in cnts], np.int32)
        features['lens'] = lens
        features['cnts'] = np.array(_pad_cnts(cnts, max(lens)), np.float32)
        features['is_text_cnts'] = True
        yield features


def get_eval_input():
    eval_dataset = tf.data.Dataset.from_generator(generator_eval,{'input_img': tf.float32,
                                                                  'lens': tf.int32,
                                                                  'cnts': tf.float32,
                                                                  'is_text_cnts': tf.bool},
                                                  {'input_img': (
                                                      tf.Dimension(None), tf.Dimension(None), tf.Dimension(None),
                                                      tf.Dimension(None)),
                                                   'lens': (tf.Dimension(None),),
                                                   'cnts': (
                                                       tf.Dimension(None), tf.Dimension(None), tf.Dimension(None),
                                                       tf.Dimension(None)),
                                                   'is_text_cnts':()}
                                                  )
    iterator = eval_dataset.make_one_shot_iterator()
    features = iterator.get_next()
    return features


if __name__ == '__main__':
    start_queue('dlakfj')
    pass
    # file_names_totaltext_train = [TOTAL_TRAIN_DIR+name for name in os.listdir(TOTAL_TRAIN_DIR)]
    #
    # for file_name in file_names_totaltext_train:
    #     print(file_name)
    #     img_name, img, cnts, maps=loading_data(file_name, False,False,False,True)
    #     print(img.shape)
    #     print(maps.shape)
