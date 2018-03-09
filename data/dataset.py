import tensorflow as tf
import os
import numpy as np
from data.data_labelling import pixellink_prepro
import multiprocessing as mp
import pickle
import gzip
import cv2

TOTAL_TRAIN_DIR = '/home/rjq/data_cleaned/pkl/totaltext_train_care/'
TOTAL_TEST_DIR = '/home/rjq/data_cleaned/pkl/totaltext_test_care/'


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
Q = mp.Queue(maxsize=2000)
print('queue excuted')


def enqueue(file_name):
    img_name, img, cnts, maps = pixellink_prepro(pickle.load(open(file_name,'rb')))
    # print(img.shape)
    # print(maps.shape)
    Q.put({'input_img': img,
           'Labels': maps.astype(np.float32)})


def start_queue(params):
    thread_num = params.thread_num
    file_names_totaltext_train = [TOTAL_TRAIN_DIR+name for name in os.listdir(TOTAL_TRAIN_DIR)] *params.epoch

    print('start')
    pool = mp.Pool(thread_num)
    pool.map_async(enqueue,file_names_totaltext_train)
    # pool.map(enqueue,file_names_totaltext_train)
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
    # train_dataset = tf.data.Dataset.from_generator(g, {'input_img':tf.float32,
    #                                                     'Labels': tf.float32},
    #                                                {'input_img': (tf.Dimension(None),tf.Dimension(None),tf.Dimension(None)),
    #                                                 'Labels': (tf.Dimension(None),tf.Dimension(None),tf.Dimension(None))}
    #                                                )
    train_dataset = tf.data.Dataset.from_generator(g, {'input_img':tf.float32,
                                                        'Labels': tf.float32},
                                                   {'input_img': (512,512,3),
                                                    'Labels': (256,256,10)}
                                                   )
    train_dataset = train_dataset.shuffle(params.shuffle_buffer).batch(params.batch_size).prefetch(params.prefetch_buffer)
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
    file_names = [TOTAL_TEST_DIR+name for name in os.listdir(TOTAL_TEST_DIR)][:20]
    for file_name in file_names:
        ins = pickle.load(open(file_name, 'rb'))
        img = ins['img']
        cnts = ins['contour']
        if img.shape[0] >= 512 or img.shape[1] >= 512:
            ratio1 = img.shape[0]/512
            ratio2 = img.shape[1]/512
            ratio = max(ratio1,ratio2)
            img = cv2.resize(img, (int(img.shape[1]/ratio), int(img.shape[0]/ratio)))
            cnts = [np.array(cnt/ratio, np.int32) for cnt in cnts]

        features = dict()
        features["input_img"] = np.expand_dims(img,0).astype(np.float32)
        lens = np.array([cnt.shape[0] for cnt in cnts], np.int32)
        features['lens'] = lens
        features['cnts'] = np.array(_pad_cnts(cnts, max(lens)), np.float32)
        features['care'] = np.array(ins['care']).astype(np.int32)
        # features['imname'] = ins['img_name']
        yield features


def get_eval_input():
    eval_dataset = tf.data.Dataset.from_generator(generator_eval,{'input_img': tf.float32,
                                                                  'lens': tf.int32,
                                                                  'cnts': tf.float32,
                                                                  'care': tf.int32},
                                                  {'input_img': (
                                                      tf.Dimension(None), tf.Dimension(None), tf.Dimension(None),
                                                      3),
                                                   'lens': (tf.Dimension(None),),
                                                   'cnts': (
                                                       tf.Dimension(None), tf.Dimension(None), tf.Dimension(None),
                                                       tf.Dimension(None)),
                                                   'care':(tf.Dimension(None),)}
                                                  )
    iterator = eval_dataset.make_one_shot_iterator()
    features = iterator.get_next()
    return features


def generator_infer():
    file_names = [TOTAL_TEST_DIR+name for name in os.listdir(TOTAL_TEST_DIR)]
    for file_name in file_names:
        ins = pickle.load(open(file_name, 'rb'))
        img = ins['img']
        cnts = ins['contour']
        if img.shape[0] >= 2000 or img.shape[1] >= 2000:
            ratio1 = img.shape[0]/2000
            ratio2 = img.shape[1]/2000
            ratio = max(ratio1,ratio2)
            img = cv2.resize(img, (int(img.shape[1]/ratio), int(img.shape[0]/ratio)))
            cnts = [np.array(cnt/ratio, np.int32) for cnt in cnts]

        features = dict()
        features["input_img"] = np.expand_dims(img,0).astype(np.float32)
        lens = np.array([cnt.shape[0] for cnt in cnts], np.int32)
        features['lens'] = lens
        features['cnts'] = np.array(_pad_cnts(cnts, max(lens)), np.float32)
        features['care'] = np.array(ins['care']).astype(np.int32)
        # features['imname'] = ins['img_name']
        yield features


def get_inference_input(params):
    eval_dataset = tf.data.Dataset.from_generator(generator_infer,{'input_img': tf.float32,
                                                                  'lens': tf.int32,
                                                                  'cnts': tf.float32,
                                                                  'care': tf.int32},
                                                  {'input_img': (
                                                      tf.Dimension(None), tf.Dimension(None), tf.Dimension(None),
                                                      3),
                                                   'lens': (tf.Dimension(None),),
                                                   'cnts': (
                                                       tf.Dimension(None), tf.Dimension(None), tf.Dimension(None),
                                                       tf.Dimension(None)),
                                                   'care':(tf.Dimension(None),)}
                                                  )
    iterator = eval_dataset.make_one_shot_iterator()
    features = iterator.get_next()
    return features


if __name__ == '__main__':
    # start_queue('dlakfj')
    pass
    # file_names_totaltext_train = [TOTAL_TRAIN_DIR+name for name in os.listdir(TOTAL_TRAIN_DIR)]
    #
    # for file_name in file_names_totaltext_train:
    #     print(file_name)
    #     img_name, img, cnts, maps=loading_data(file_name, False,False,False,True)
    #     print(img.shape)
    #     print(maps.shape)
