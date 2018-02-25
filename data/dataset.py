import tensorflow as tf
import os
import numpy as np
from data.data_augmentation import DataAugmentor
from data.data_labelling import data_churn
import multiprocessing as mp
import pickle

PKL_DIR = '/home/rjq/data_cleaned/pkl/'
TOTAL_TRAIN_DIR = 'totaltext_train/'
TOTAL_TEST_DIR = 'totaltext_test/'
SYN_DIR = '/media/sda/eccv2018/data/pkl/result2'

DA = DataAugmentor()
labelling = data_churn()


def _load_file(file):
    return pickle.load(open(file, 'rb'))


def _data_aug(ins, augment_rate, test_mode=False, real_test=False):
    return DA.augment(ins, augment_rate=augment_rate, test_mode=test_mode, real_test=real_test)


def _data_label(ins):
    return labelling._data_labeling(ins['img_name'], ins['img'],
                                    ins['contour'], ins['is_text_cnts'],
                                    ins['left_top'], ins['right_bottom'],
                                    ins.get('chars', None))


def loading_data(file, test_mode=False, real_test=False):
    return _data_label(_data_aug(_load_file(file), augment_rate=100, test_mode=test_mode, real_test=real_test))


def decompress(ins):
    name = ins[0]
    img = ins[1]
    non_zero, radius, cos, sin = ins[2]
    maps = np.zeros(shape=(*(img.shape[:2]), 5))
    maps[:, :, 4] = np.cast['uint8'](ins[3])  # -->TR
    maps[:, :, 0][non_zero] = 1               # -->TCL
    maps[:, :, 1][non_zero] = radius          # -->radius
    maps[:, :, 2][non_zero] = cos             # -->cos
    maps[:, :, 3][non_zero] = sin             # -->TCL
    cnt = ins[4]
    return (name, img, maps, cnt)

total_q = mp.Queue(maxsize=3000)
syn_q = mp.Queue(maxsize=3000)
print('queue excuted')


def enqueue_total(file_name, test_mode=False, real_test=False, if_decompress=False):
    if not if_decompress:
        img_name, img, maps, cnts = _data_label(_data_aug(pickle.load(open(file_name, 'rb')), 100, test_mode, real_test))
    else:
        img_name, img, maps, cnts = _data_label(_data_aug(decompress(pickle.load(open(file_name, 'rb'))), 100, test_mode, real_test))
    total_q.put({'input_img': img,
                'Labels': maps.astype(np.float32)})


def enqueue_syn(file_name, test_mode=False, real_test=False, if_decompress=False):
    if not if_decompress:
        img_name, img, maps, cnts = _data_label(_data_aug(pickle.load(open(file_name, 'rb')), 100, test_mode, real_test))
    else:
        img_name, img, maps, cnts = _data_label(_data_aug(decompress(pickle.load(open(file_name, 'rb'))), 100, test_mode, real_test))
    syn_q.put({'input_img': img,
                'Labels': maps.astype(np.float32)})


def start_queue(params):
    thread_num = params.thread_num
    file_names_total = [PKL_DIR+TOTAL_TRAIN_DIR+name for name in os.listdir(PKL_DIR+TOTAL_TRAIN_DIR)]
    file_names_syn = []
    for name in os.listdir(SYN_DIR):
        if '.gz' not in name:
            file_names_syn.append(SYN_DIR+name)

    print('start')
    pool1 = mp.Pool(thread_num)
    for file_name in file_names_syn:
        pool1.apply_async(enqueue_syn, (file_name, False, False, False))

    # pool2 = mp.Pool(thread_num)
    # for file_name in file_names_total:
    #     pool2.apply_async(enqueue_total, (file_name, True, False, True))
    print('end')


def get_generator(aqueue):
    def func():
        while True:
            features = aqueue.get()

            yield {'input_img': features['input_img'].astype(np.float32),
                    'Labels': features['Labels'].astype(np.float32)}
    return func


def get_train_input(params):
    syn_g = get_generator(syn_q)
    syn_dataset = tf.data.Dataset.from_generator(syn_g, {'input_img':tf.float32,
                                                        'Labels': tf.float32},
                                                   {'input_img': (tf.Dimension(None),tf.Dimension(None),tf.Dimension(None)),
                                                    'Labels': (tf.Dimension(None),tf.Dimension(None),tf.Dimension(None))}
                                                   )
    syn_dataset = syn_dataset.repeat(params.pre_epoch).batch(params.batch_size)

    # total_g = get_generator(total_q)
    # total_dataset = tf.data.Dataset.from_generator(total_g, {'input_img':tf.float32,
    #                                                     'Labels': tf.float32},
    #                                                {'input_img': (tf.Dimension(None),tf.Dimension(None),tf.Dimension(None)),
    #                                                 'Labels': (tf.Dimension(None),tf.Dimension(None),tf.Dimension(None))}
    #                                                )
    # total_dataset = total_dataset.batch(params.batch_size).repeat()
    # train_dataset = syn_dataset.concatenate(total_dataset)
    train_dataset = syn_dataset
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
    file_names = [PKL_DIR+TOTAL_TEST_DIR+name for name in os.listdir(PKL_DIR+TOTAL_TEST_DIR)]
    for file_name in file_names[:2]:
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