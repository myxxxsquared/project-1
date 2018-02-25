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


q = mp.Queue(maxsize=3000)
print('queue excuted')


def enqueue(file_name):
    img_name, img, maps, cnts = loading_data(PKL_DIR+file_name)
    q.put({'input_img': img,
           'Labels': maps.astype(np.float32)})


def start_queue(params):
    thread_num = params.thread_num
    epoch = params.epoch
    file_names = [TOTAL_TRAIN_DIR+name for name in os.listdir(PKL_DIR+TOTAL_TRAIN_DIR)]*epoch

    print('start')
    pool = mp.Pool(thread_num)
    for file_name in file_names:
        pool.apply_async(enqueue, (file_name,))
    print('end')


def generator(params, aqueue):
    while True:
        imgs = []
        mapss = []
        for i in range(params.batch_size):
            features = aqueue.get()
            img = features['input_img']
            maps = features['Labels']
            imgs.append(np.expand_dims(img,0))
            mapss.append(np.expand_dims(maps,0))

        yield {'input_img': np.concatenate(imgs).astype(np.float32),
                'Labels': np.concatenate(mapss).astype(np.float32)}


def get_train_input(params):
    iterator = generator(params, q)
    return iterator.__next__()


def generator_eval():
    file_names = [PKL_DIR+TOTAL_TEST_DIR+name for name in os.listdir(PKL_DIR+TOTAL_TEST_DIR)]
    for file_name in file_names[:2]:
        img_name, img, maps, cnts = loading_data(file_name, True, False)

        features = {}
        features["input_img"] = np.expand_dims(img,0).astype(np.float32)
        features["cnts"] = [cnt.astype(np.float32).tolist() for cnt in cnts]
        features['is_text_cnts'] = True
        yield features


def get_eval_input():
    iterator = generator_eval()
    return iterator.__next__()


if __name__ == '__main__':
    for i in range(500):
        res = get_eval_input()
        print(res['input_img'].shape)