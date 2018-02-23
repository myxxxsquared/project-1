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


def get_train_input(params):
    q = mp.Queue()

    # def job(q, start, end):
    #     for i in range(start, end):
    #         q.put(loading_data(PKL_DIR+str(i)+'.bin'))

    def job(q, start, end):
        for i in range(start, end):
            q.put({'input_img': np.ones((12, 512,512, 3)).astype(np.float32),
                   'Labels': np.ones((12, 512,512,5)).astype(np.float32)})

    starts = [0, 100, 200]
    ends = [100, 200, 300]

    jobs = []
    for i in range(3):
        jobs.append(mp.Process(target=job, args=(q, starts[i], ends[i])))
    for i in range(3):
        jobs[i].start()
    print(q.get())
    print('get one example')
    # for i in range(3):
    #     jobs[i].join()
    print('end')

    def generator(q):
        while True:
            # with tf.device('/cpu:0'):
            #     features = {'input_img': tf.convert_to_tensor(np.ones((12, 512,512, 3)).astype(np.float32)),
            #                 'Labels': tf.convert_to_tensor(np.ones((12, 512,512,5)).astype(np.float32))}
            yield q.get()

    return  generator().__next__()

if __name__ == '__main__':
    get_train_input('sdkfa')