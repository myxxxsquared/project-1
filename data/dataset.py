import tensorflow as tf
import os
import numpy as np
from data.data_augmentation import DataAugmentor
from data.data_labelling import data_churn
import multiprocessing as mp
import pickle

PKL_DIR = '/home/rjq/data_cleaned/pkl/totaltext_train/'

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


q = mp.Queue()
print('excuted')

def start_queue(params):
    thread_num = params.thread_num
    epoch = params.epoch

    def enqueue(q, file_name):
        img_name, img, maps, cnts = loading_data(file_name)
        q.put({'input_img': img,
               'Labels': maps.astype(np.float32)})
        print('example'+str(i))

    file_names = os.listdir(PKL_DIR)*epoch
    qs = [q]*len(file_names)
    args = []
    for queue, name in zip(qs, file_names):
        args.append((queue,name))

    p = mp.Pool(thread_num)
    print('start')
    p.map(enqueue, args)
    print('end')

def generator(q):
    while True:
        yield q.get()


def get_train_input(params):
    imgs = []
    mapss = []
    for i in range(params.batch_size):
        features = generator(q).__next__()
        img = features['input_img']
        map = features['Lables']
        imgs.append(img)
        mapss.append(map)
    return {'input_img': np.concatenate(imgs).astype(np.float32),
                   'Labels': np.concatenate(mapss).astype(np.float32)}

if __name__ == '__main__':
    # res = loading_data(PKL_DIR+'100.bin')
    # print(len(res))
    for i in range(500):
        print(get_train_input('x'))