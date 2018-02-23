import os
import time
import random
import glob
import pickle
import numpy as np
from random import shuffle
from multiprocessing import Pool
from .data_augmentation import DataAugmentor
from .data_labelling import data_churn
from .myqueue import MyQueue
from multiprocessing import Process


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


def MultiRun(func, args, thread_num=32):
    """
    runs the specified functions (func, args) with a Pool size of thread_num
    :param funcs: List(the target function)
    :param args: List[args in Dict], or a single arg for next
    :param thread_num: 32 by default
    :return:
    """

    p = Pool(thread_num)
    result = []
    for i in range(len(args)):
        result.append(p.apply_async(func, args=args[i]))
    p.close()
    p.join()

    return [r.get() for r in result]


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


def load_pre_gen(file):
    return decompress(pickle.load(open(file, 'rb')))


class data_batch(object):
    def __init__(self, configs, logs):
        self.configs = configs
        self.logs = logs
        self.test_data = {'resized_test_set': [], 'original_size_test_set': []}

    def _data_test_set_loader(self):
        name = self.configs.dataset_name + '_' + 'test'
        data_path = glob.glob(os.path.join(self.configs.raw_data_path, name, '*.bin'))

        data_output = MultiRun(loading_data,
                               [(file, True, False) for file in
                                data_path],  # 32
                               self.configs.threads)

        for data_ins in data_output:
            names = data_ins[0]
            images = data_ins[1]
            maps = data_ins[2]
            cnt = data_ins[3]
            self.test_data['resized_test_set'].append((names, images, maps, cnt))

        self.logs['info']('resized_test_set loaded.')

        data_output = MultiRun(loading_data,
                               [(file, True, True) for file in
                                data_path],  # 32
                               self.configs.threads)

        for data_ins in data_output:
            names = data_ins[0]
            images = data_ins[1]
            maps = data_ins[2]
            cnt = data_ins[3]
            self.test_data['original_size_test_set'].append((names, images, maps, cnt))

        self.logs['info']('original_size_test_set loaded.')

    def data_test_set_gen(self, real_test):
        """
        :param task:task="train","dev","test","synth"
        :return:
        """
        if len(self.test_data['resized_test_set']) == 0:
            self._data_test_set_loader()

        if not real_test:
            for i in range((len(self.test_data['resized_test_set']) - 1) // self.configs.batch_size + 1):
                names = [ins[0] for ins in self.test_data['resized_test_set'][i * self.configs.batch_size:(i + 1) * self.configs.batch_size]]
                images = np.stack([ins[1] for ins in self.test_data['resized_test_set'][
                    i * self.configs.batch_size:(i + 1) * self.configs.batch_size]], axis=0)
                maps = np.stack([ins[2] for ins in self.test_data['resized_test_set'][
                    i * self.configs.batch_size:(i + 1) * self.configs.batch_size]], axis=0)
                cnts = [ins[3] for ins in self.test_data['resized_test_set'][
                    i * self.configs.batch_size:(i + 1) * self.configs.batch_size]]
                yield names, images, maps, cnts
        else:
            for ins in self.test_data['original_size_test_set']:
                names = [ins[0]] * len(self.configs.gpu_list)
                images = np.stack([ins[1]] * len(self.configs.gpu_list), axis=0)
                maps = np.stack([ins[2]] * len(self.configs.gpu_list), axis=0)
                cnts = [ins[3]] * len(self.configs.gpu_list)
                yield names, images, maps, cnts

    def data_gen_multiprocess(self, task):
        # print("data_gen_multiprocess")

        queue = MyQueue(maxsize=self.configs.QueueLength)

        def generate_work(id):
            #print("generate_work", id)
            random.seed(int(time.time() * 1000) + id)
            #print("generate_work started", id)

            if task == 'Synth':
                name = 'synthtext_chars'
            else:
                name = self.configs.dataset_name + '_' + task
            data_path = glob.glob(os.path.join(self.configs.raw_data_path, name, '*.bin'))

            shuffle(data_path)
            p = 0

            while True:
                t = time.time()
                if task != 'test' and p + self.configs.batch_size > len(data_path):
                    p = 0
                    shuffle(data_path)
                if p >= len(data_path):
                    break
                data_output = [loading_data(file, task == 'test', self.configs.real_test)
                               for file in data_path[p:p + self.configs.batch_size]]
                for i in range(len(data_output) - 1, -1, -1):
                    if data_output[i][0] is None:
                        data_output.pop(i)
                while len(data_output) % len(self.configs.gpu_list) != 0:
                    data_output.pop(0)
                p += self.configs.batch_size
                names = [ins[0] for ins in data_output]
                images = np.stack([ins[1] for ins in data_output], axis=0)
                maps = np.stack([ins[2] for ins in data_output], axis=0)
                queue.put((names,
                           images,
                           maps,
                           data_output[0][3]),
                          os.path.join(self.configs.pre_labelled_data_path, name))
                print(time.time() - t)

        def load_work(id):
            #print("generate_work", id)
            random.seed(int(time.time() * 1000) + id)
            #print("generate_work started", id)

            if task == 'Synth':
                name = 'synthtext_chars'
            else:
                name = self.configs.dataset_name + '_' + task
            data_path = glob.glob(os.path.join(self.configs.pre_labelled_data_path, name, '*'))

            shuffle(data_path)
            p = 0

            while True:
                t = time.time()
                if task != 'test' and p + self.configs.batch_size > len(data_path):
                    p = 0
                    shuffle(data_path)
                if p >= len(data_path):
                    break
                data_output = [load_pre_gen(file)
                               for file in data_path[p:p + self.configs.batch_size]]
                for i in range(len(data_output) - 1, -1, -1):
                    if data_output[i][0] is None:
                        data_output.pop(i)
                while len(data_output) % len(self.configs.gpu_list) != 0:
                    data_output.pop(0)
                p += self.configs.batch_size
                names = [ins[0] for ins in data_output]
                images = np.stack([ins[1] for ins in data_output], axis=0)
                maps = np.stack([ins[2] for ins in data_output], axis=0)
                queue.put((names,
                           images,
                           maps,
                           data_output[0][3]),
                          None)
                print(time.time() - t)

        # pool = Pool(30)
        for i in range(self.configs.threads):  # it takes at least 4 processes for Synth, and more for Totaltext(augmentation)
            if self.configs.use_pre_generated_data:
                print('load from pre-generated files')
                process = Process(target=load_work, args=(i,))
            else:
                print('generate data and save')
                process = Process(target=generate_work, args=(i,))
            process.daemon = True
            process.start()
            # pool.apply_async(generate_work, (i))
        # pool.close()
        print('Background Multi-processing for Data Pre-Processing has started.')
        return queue
