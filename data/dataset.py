import tensorflow as tf
import os
import numpy as np
from data.data_augmentation import DataAugmentor
from data.data_labelling import data_churn
import multiprocessing as mp
import pickle
import gzip

TOTAL_TRAIN_DIR = '/home/rjq/data_cleaned/pkl/totaltext_train/'
TOTAL_TEST_DIR = '/home/rjq/data_cleaned/pkl/totaltext_test/'
SYN_DIR = '/media/sda/eccv2018/data/pkl/result2/'

SYN = '/home/lsb/pre_labelled_data/synthtext_chars/'
TOTAL_TRAIN = '/home/lsb/pre_labelled_data/totaltext_train/'


DA = DataAugmentor()
labelling = data_churn()


def _load_file(file, syn):
    if not syn:
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


def _data_label(ins):
    try:
        data = labelling._data_labeling(ins['img_name'], ins['img'],
                                        ins['contour'], ins['is_text_cnts'],
                                        ins['left_top'], ins['right_bottom'],
                                        ins.get('chars', None))
        return data
    except:
        return None, None, None, None


def loading_data(file, test_mode=False, real_test=False, syn=True):
    return _data_label(_data_aug(_load_file(file, syn=syn), augment_rate=1, test_mode=test_mode, real_test=real_test))


q = mp.Queue(maxsize=3000)
print('queue excuted')


def enqueue(file_name, test_mode, real_test, syn):
    img_name, img, maps, cnts = loading_data(file_name, test_mode, real_test, syn)
    q.put({'input_img': img,
           'Labels': maps.astype(np.float32)})


def start_queue(params):
    thread_num = params.thread_num
    flag = []
    file_names_syn = [SYN_DIR+name for name in os.listdir(SYN_DIR)]*params.pre_epoch
    for _ in range(len(file_names_syn)):
        flag.append(True)
    file_names_total = [TOTAL_TRAIN_DIR+name for name in os.listdir(TOTAL_TRAIN_DIR)]*params.epoch
    for _ in range(len(file_names_total)):
        flag.append(False)
    file_names = file_names_syn+file_names_total
    print('start')
    pool = mp.Pool(thread_num)
    for file_name, f in zip(file_names,flag):
        pool.apply_async(enqueue, (file_name, False, False, f))
    print('end')



# def get_generator(params, aqueue):
#     def func():
#         while True:
#             imgs = []
#             mapss = []
#             for i in range(params.batch_size):
#                 features = aqueue.get()
#                 img = features['input_img']
#                 maps = features['Labels']
#                 imgs.append(np.expand_dims(img,0))
#                 mapss.append(np.expand_dims(maps,0))
#
#             yield {'input_img': np.concatenate(imgs).astype(np.float32),
#                     'Labels': np.concatenate(mapss).astype(np.float32)}
#     return func
def get_generator(params, aqueue):
    def func():
        while True:
            features = aqueue.get()

            yield {'input_img': features['input_img'].astype(np.float32),
                    'Labels': features['Labels'].astype(np.float32)}
    return func


def get_train_input(params):
    g = get_generator(params, q)
    train_dataset = tf.data.Dataset.from_generator(g, {'input_img':tf.float32,
                                                        'Labels': tf.float32},
                                                   {'input_img': (tf.Dimension(None),tf.Dimension(None),tf.Dimension(None)),
                                                    'Labels': (tf.Dimension(None),tf.Dimension(None),tf.Dimension(None))}
                                                   )
    # train_dataset = train_dataset.shuffle(params.suffle_buffer)
    train_dataset = train_dataset.batch(params.batch_size)
    iterator = train_dataset.make_one_shot_iterator()
    features = iterator.get_next()
    return features


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
    for i in range(500):
        res = get_eval_input()
        print(res['input_img'].shape)