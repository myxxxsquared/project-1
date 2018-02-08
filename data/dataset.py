import tensorflow as tf
import pickle
import numpy as np
from .data_augementation import DataAugmentor
from .data_labeling import data_churn







def load_file(file):
    return pickle.load(open(file,'rb'))


def data_aug(ins,augment_rate):
    DA = DataAugmentor()
    return DA.augment(ins,augment_rate=augment_rate)


def data_label(ins):
    labelling = data_churn()
    return labelling._data_labeling(ins['img_name'],ins['img'],
                                    ins['contour'],ins['is_text_cnts'],
                                    ins['left_top'],ins['right_bottom'],
                                    ins.get('chars',None))


def syn_wrapper(index):
    # PKL_DIR = '/home/rjq/data_cleaned/pkl/'
    # file = PKL_DIR + 'totaltext_train/' + str(index) + '.bin'
    # img_name, img, maps = data_label(data_aug(load_file(file), augment_rate=100))
    # # img_name, img, maps = data_label(data_aug(load_file(file), augment_rate=100))
    # [TR, TCL, radius, cos_theta, sin_theta] = maps
    import time
    time.sleep(5)
    img = np.zeros((512,512,3))
    TR = np.zeros((512,512,1))
    TCL = np.zeros((512,512,1))
    radius = np.zeros((512,512,1))
    cos_theta = np.zeros((512,512,1))
    sin_theta = np.zeros((512,512,1))
    img = np.reshape(np.array(img, np.float32),(512,512,3))
    TR = np.reshape(np.array(TR, np.float32),(512,512,1))
    TCL = np.reshape(np.array(TCL, np.float32),(512,512,1))
    radius = np.reshape(np.array(radius, np.float32),(512,512,1))
    cos_theta = np.reshape(np.array(cos_theta, np.float32),(512,512,1))
    sin_theta = np.reshape(np.array(sin_theta, np.float32),(512,512,1))

    # img = tf.convert_to_tensor(np.reshape(np.array(img, np.float32),(512,512,3)))
    # TR = tf.convert_to_tensor(np.reshape(np.array(TR, np.float32),(512,512,1)))
    # TCL = tf.convert_to_tensor(np.reshape(np.array(TCL, np.float32),(512,512,1)))
    # radius = tf.convert_to_tensor(np.reshape(np.array(radius, np.float32),(512,512,1)))
    # cos_theta = tf.convert_to_tensor(np.reshape(np.array(cos_theta, np.float32),(512,512,1)))
    # sin_theta = tf.convert_to_tensor(np.reshape(np.array(sin_theta, np.float32),(512,512,1)))

    # print(img.shape)
    # res = np.stack((img, TR, TCL, radius, cos_theta, sin_theta))
    return img, TR, TCL, radius, cos_theta, sin_theta


BUFFER_SIZE=3000
def get_train_input(params):
    # syn_dataset = tf.data.Dataset.range(858749+1).repeat(params.pretrain_num)
    #syn_dataset=tf.contrib.data.python.ops.dataset_ops.Dataset.range(1000).repeat(params.pretrain_num)
    syn_dataset = tf.data.Dataset.range(1000).repeat(params.pretrain_num)
    # syn_dataset = tf.data.Dataset.range(1000)

    syn_dataset = syn_dataset.map(
        lambda index: tuple(tf.py_func(
            syn_wrapper, [index], [tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32])),
        num_parallel_calls=40).prefetch(BUFFER_SIZE)

    syn_dataset = syn_dataset.batch(32).prefetch(1000)

    # syn_dataset.map_and_batch(lambda index: tuple(tf.py_func(
    #         syn_wrapper, [index], [tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32])),
    #                           32, 40)
    #

    iterator = syn_dataset.make_one_shot_iterator()



    features = iterator.get_next()
    # queue = tf.FIFOQueue(100000, dtypes=[tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32],
    #                      shapes=[(512,512,3),(512,512,1),(512,512,1),(512,512,1),(512,512,1),(512,512,1)])
    # enqueue_op = queue.enqueue(features)
    # qr = tf.train.QueueRunner(queue, [enqueue_op] * 80)
    # tf.train.add_queue_runner(qr)
    # inputs = queue.dequeue_many(32)

    # Launch the graph.
    # sess = tf.Session()
    # # Create a coordinator, launch the queue runner threads.
    # coord = tf.train.Coordinator()
    # enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
    # # Run the training loop, controlling termination with the coordinator.
    # for step in range(1000000):
    #     if coord.should_stop():
    #         break
    #     sess.run(train_op)
    # # When done, ask the threads to stop.
    # # coord.request_stop()
    # # # And wait for them to actually do it.
    # # coord.join(enqueue_threads)
    #
    # # syn_dataset = syn_dataset.batch(32)


    return features



