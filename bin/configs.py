import tensorflow as tf
import os


class configs(object):
    def __init__(self):
        tf.app.flags.DEFINE_integer("batch_size", 16, "batch_size per gpu")
        tf.app.flags.DEFINE_string('raw_data_path',
                                   '/home/lsb/data_cleaned/pkl/',
                                   'raw_data_path')
        tf.app.flags.DEFINE_string('dataset_name','totaltext','dataset_name')
        tf.app.flags.DEFINE_string('SynthPath',
                                   '/home/lsb/data_cleaned/pkl/synthtext',
                                   'SynthPath')
        tf.app.flags.DEFINE_string("epochs", '1,20', "epochs")
        tf.app.flags.DEFINE_string('dataset_size', '1255/300/850000', 'dataset_size')#833019
        tf.app.flags.DEFINE_string('tfrecord_path', 'tfrecord', 'tfrecord_path')
        tf.app.flags.DEFINE_string('weights_path', '~/pre_trained', 'weights_path')
        tf.app.flags.DEFINE_string("input_size", '512*512*3', "input_size")  # =image_size
        tf.app.flags.DEFINE_string("Label_size", '512*512*5', "Label_size")
        tf.app.flags.DEFINE_string("padding", 'SAME', "padding")
        tf.app.flags.DEFINE_string("pooling", 'max', "pooling")  # max
        tf.app.flags.DEFINE_string("basenet", 'vgg16', "basenet")
        tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning_rate")
        tf.app.flags.DEFINE_string("save_path", 'model_backup/', "save_path")
        tf.app.flags.DEFINE_float("momentum", 0.9, "momentum")
        tf.app.flags.DEFINE_string("log_path", '', "log_path")
        tf.app.flags.DEFINE_string("restore_path", '', "restore_path")
        tf.app.flags.DEFINE_string("test_path", 'test_output', "test_path")
        tf.app.flags.DEFINE_string('record_path',
                                   '/home/lsb/runninglog',
                                   'record')
        tf.app.flags.DEFINE_string('code_name', 'debug', 'code_name')
        tf.app.flags.DEFINE_boolean('data_aug', True, 'data_aug')
        tf.app.flags.DEFINE_boolean('data_aug_noise', True, 'data_aug_noise')
        tf.app.flags.DEFINE_string("US_Params", '3 3 2 2 same ReLU', "US_Params")
        tf.app.flags.DEFINE_string("predict_channels", '128 64 32 32', "predict_channels")
        tf.app.flags.DEFINE_string("upsampling", 'DeConv', "upsampling")
        tf.app.flags.DEFINE_integer("Predict_stage", 4, "Predict_stage")
        tf.app.flags.DEFINE_string("optimizer", 'Adam', "optimizer")#'Adam', 'YF'
        tf.app.flags.DEFINE_string('gpu_list','0','gpu_list')#'0,1,2,3,4,...'
        tf.app.flags.DEFINE_boolean('multi_gpu_switch', False, 'multi_gpu_switch')
        self.FLAGS = tf.app.flags.FLAGS
        self._read()

    def _read(self):
        self.record_path = self.FLAGS.record_path
        self.SynthPath=self.FLAGS.SynthPath
        self.dataset_name=self.FLAGS.dataset_name
        self.tfrecord_path = os.path.join(self.FLAGS.raw_data_path, self.FLAGS.tfrecord_path)
        self.raw_data_path = self.FLAGS.raw_data_path
        self.epochs = list(map(int,self.FLAGS.epochs.split(',')))
        self.epochs={'Synth':self.epochs[0],'train':self.epochs[1]}
        # self.tfrecord_path = os.path.join(self.record_path,self.FLAGS.tfrecord_path)
        self.weights_path = self.FLAGS.weights_path
        self.input_size = self.image_size = tuple(map(int, self.FLAGS.input_size.split('*')))
        self.Label_size = tuple(map(int,self.FLAGS.Label_size.split('*')))
        self.padding = self.FLAGS.padding
        self.basenet = self.FLAGS.basenet
        self.pooling = self.FLAGS.pooling
        self.learning_rate = self.FLAGS.learning_rate
        self.momentum = self.FLAGS.momentum
        self.dataset_size= list(map(int,self.FLAGS.dataset_size.split('/')))
        self.size = {'train': self.dataset_size[0], 'test': self.dataset_size[1], "Synth": self.dataset_size[2]}
        self.save_path = os.path.join(self.record_path, self.FLAGS.save_path)
        self.log_path = os.path.join(self.record_path, self.FLAGS.log_path)
        self.restore_path = self.FLAGS.restore_path
        self.summary_path_directory = '/'.join(self.log_path.split('/')[:-1])
        self.test_path = os.path.join(self.record_path, self.FLAGS.test_path)
        self.code_name = self.FLAGS.code_name
        self.data_aug = self.FLAGS.data_aug
        self.data_aug_noise = self.FLAGS.data_aug_noise
        self.US_Params=self.FLAGS.US_Params
        self.upsampling=self.FLAGS.upsampling
        self.predict_channels=list(map(int,self.FLAGS.predict_channels.split()))
        self.optimizer=self.FLAGS.optimizer
        self.Predict_stage=self.FLAGS.Predict_stage
        self.gpu_list=self.FLAGS.gpu_list.split(',')
        self.multi_gpu_switch=self.FLAGS.multi_gpu_switch
        if (not self.multi_gpu_switch) and len(self.gpu_list)>1:
            raise ValueError('now using multiple gpus in single gpu mode')
        os.environ["CUDA_VISIBLE_DEVICES"] = self.FLAGS.gpu_list
        self.batch_size=self.FLAGS.batch_size*len(self.gpu_list)
        for path in [self.record_path, self.save_path, self.log_path, self.summary_path_directory,
                     self.test_path]:
            if not os.path.isdir(path):
                os.makedirs(path)

