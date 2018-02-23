import tensorflow as tf
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

def enqueue(q, start, end):
    for i in range(start, end,2):
        imgs = []
        mapss = []
        for j in range(2):
            img_name, img, maps, cnts = loading_data(PKL_DIR+str(i+j)+'.bin')
            imgs.append(img)
            mapss.append(maps)
        q.put({'input_img': np.concatenate(imgs).astype(np.float32),
               'Labels': np.concatenate(mapss).astype(np.float32)})
        print(np.concatenate(imgs).shape)
        print(np.concatenate(mapss).shape)
        # q.put({'input_img': np.ones((12, 512,512, 3)).astype(np.float32),
        #        'Labels': np.ones((12, 512,512,5)).astype(np.float32)})
        # q.put(i)

starts = [0, 100, 200]
ends = [100, 200, 300]

jobs = []
for i in range(3):
    jobs.append(mp.Process(target=enqueue, args=(q, starts[i], ends[i])))
for i in range(3):
    jobs[i].start()


def generator(q):
    while True:
        yield q.get()


def get_train_input(params):
    return generator(q).__next__()

if __name__ == '__main__':
    # res = loading_data(PKL_DIR+'100.bin')
    # print(len(res))
    for i in range(500):
        print(get_train_input('x'))