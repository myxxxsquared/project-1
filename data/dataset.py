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

thread_num = 10
batch_size = 10
q = mp.Queue()

def enqueue(q, start, end, batch_size):
    for i in range(start, end, batch_size):
        imgs = []
        mapss = []
        for j in range(batch_size):
            img_name, img, maps, cnts = loading_data(PKL_DIR+str(i+j)+'.bin')
            imgs.append(np.expand_dims(img,0))
            mapss.append(np.expand_dims(maps,0))
        q.put({'input_img': np.concatenate(imgs).astype(np.float32),
               'Labels': np.concatenate(mapss).astype(np.float32)})
        print(np.concatenate(imgs).shape)
        print(np.concatenate(mapss).shape)
        # q.put({'input_img': np.ones((12, 512,512, 3)).astype(np.float32),
        #        'Labels': np.ones((12, 512,512,5)).astype(np.float32)})
        # q.put(i)

starts = np.array(list(range(thread_num)))*(1254//thread_num+1)
ends = (np.array(list(range(thread_num)))+1)*(1254//thread_num+1)
ends[-1] = 1254

jobs = []
for i in range(thread_num):
    jobs.append(mp.Process(target=enqueue, args=(q, starts[i], ends[i], batch_size)))
for i in range(thread_num):
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