
import copy
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import xavier_initializer_conv2d as xavier


def cpu_variable_getter(getter, *args, **kwargs):
    with tf.device("/cpu:0"):
        return getter(*args, **kwargs)

"""
输入:
features['img'] np.array(shape=(batch_size, height, width, 3), dtype=tf.uint8)
features['maps'] np.array(shape=(batch_size, height_, width_, 9), dtype=tf.float32)
features['weights'] np.array(shape=(batch_size, height_, width_), dtype=tf.float32)

图片大小：
height_ = height / 2 或 height / 4
height_ = width / 2 或 width / 4
"""

class PixelLinkNetwork:
    def conv2d(self, input, shape, name):
        return tf.nn.conv2d(
            input,
            tf.get_variable(
                name+'_kernel', dtype=tf.float32, shape=shape, initializer=xavier(), trainable=True),
            strides=(1, 1, 1, 1),
            padding="SAME",
            name=name)

    def pool(self, input, stride, name):
        return tf.nn.max_pool(
            input,
            ksize=(1, 2, 2, 1),
            strides=stride,
            padding="SAME",
            name=name)

    def pre_processing(self, image):
        return (image - np.reshape((123.68, 116.78, 103.94), (1, 1, 1, 3))) / 256

    def vgg_base(self, input_image):
        layer = [
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',  # part1
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',  # part2
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
            'relu3_3', 'pool3',                                   # part3
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
            'relu4_3', 'pool4',                                   # part4
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
            'relu5_3', 'pool5',                                   # part5
            'conv6_1', 'relu6_1',  'conv6_2', 'relu6_2'           # part6
        ]

        filter_shapes = {
            # part1
            'conv1_1': (3, 3, 3, 64), 'conv1_2': (3, 3, 64, 64),
            # part2
            'conv2_1': (3, 3, 64, 128), 'conv2_2': (3, 3, 128, 128),
            # part3
            'conv3_1': (3, 3, 128, 256), 'conv3_2': (3, 3, 256, 256), 'conv3_3': (3, 3, 256, 256),
            # part4
            'conv4_1': (3, 3, 256, 512), 'conv4_2': (3, 3, 512, 512), 'conv4_3': (3, 3, 512, 512),
            # part4
            'conv5_1': (3, 3, 512, 512), 'conv5_2': (3, 3, 512, 512), 'conv5_3': (3, 3, 512, 512),
            # part 6
            'conv6_1': (3, 3, 512, 512), 'conv6_2': (3, 3, 512, 512)
        }

        pool_strides = {
            'pool1': (1, 2, 2, 1),
            'pool2': (1, 2, 2, 1),
            'pool3': (1, 2, 2, 1),
            'pool4': (1, 2, 2, 1),
            'pool5': (1, 1, 1, 1),
        }

        net = {}

        activations = self.pre_processing(input_image)
        for name in layer:
            Layer_type = name[:4]
            if Layer_type == 'conv':
                activations = self.conv2d(
                    activations, filter_shapes[name], name)
            elif Layer_type == 'relu':
                activations = tf.nn.relu(activations)
            elif Layer_type == 'pool':
                activations = self.pool(activations, pool_strides[name], name)
            net[name] = activations

        return net

    def prediction(self, vgg):
        maps = [vgg[x] for x in
                {
                    2: ['relu2_2', 'relu3_3', 'relu4_3', 'relu5_3', 'relu6_2'],
                    4: ['relu3_3', 'relu4_3', 'relu5_3', 'relu6_2']}
                [self.parameters.output_scalar][::-1]]

        with tf.variable_scope('T'):
            T = self.prediction_block(maps, 2)
        with tf.variable_scope('L'):
            L = self.prediction_block(maps, 16)
        return tf.concat([T, L], axis=3)

    def prediction_block(self, maps, ochannels):
        prediction = self.conv2d(
            maps[0], (1, 1, maps[0].shape[3], ochannels), 'conv_0')
        for i in range(1, len(maps)):
            ff = maps[i]
            dynamic_shape = tf.shape(ff)
            ffsize = (ff.shape[1], ff.shape[2])
            if not ffsize[0] or not ffsize[1]:
                ffsize = (dynamic_shape[1], dynamic_shape[2])
            prediction = tf.image.resize_images(prediction, ffsize)  \
                + self.conv2d(maps[i], (1, 1, maps[i].shape[3],
                                        ochannels), 'conv_%d' % (i,))
        return self.conv2d(prediction, (1, 1, ochannels, ochannels), 'conv_o')

    def build_loss(self, prediction, maps, weights):
        """
        maps: [text, links ... ]
        0 1 2
        3 X 4
        5 6 7
        """

        # NOTE: config r, lambda
        r = tf.constant(3, dtype=tf.float32, name='param_r')
        lambda_ = tf.constant(2, dtype=tf.float32, name='param_lambda')

        prediction = tf.reshape(prediction, (-1, 18))
        maps = tf.reshape(maps, (-1, 9))
        weights = tf.reshape(weights, (-1))

        cross_entropy = []
        for i in range(9):
            cross_entropy.append(tf.losses.softmax_cross_entropy(
                tf.concat([1 - maps[:, i:i+1], maps[:, i:i+1]], axis=1), prediction[:, 2*i:2*i+1], reduction=tf.losses.Reduction.NONE))

        with tf.name_scope('T'):
            pos_region = maps[:, 0]
            neg_region = 1 - pos_region
            posnum = tf.reduce_sum(pos_region) + 1e-5
            negsum = tf.reduce_sum(neg_region) + 1e-5
            k = tf.cast(tf.reduce_min(
                (r*posnum + 1, tf.int32, negsum)), tf.int32).values
            weighted_loss = cross_entropy[0] * weights
            pos_loss = pos_region * weighted_loss
            neg_loss = neg_region * weighted_loss
            T_loss = (tf.reduce_sum(pos_loss) +
                      tf.reduce_sum(tf.nn.top_k(neg_loss, k=k))) / (1 + r)
            T_loss = lambda_ * T_loss

        with tf.name_scope('L'):
            pos_region = maps[:, 1:9]
            neg_region = 1 - pos_region
            link_loss= tf.concat([tf.expand_dims(x, 1) for x in cross_entropy[1:9]], 1)
            weights = tf.reshape(weights, (-1, 1))
            pos_weights = pos_region * weights
            neg_weights = neg_region * weights
            L_loss = tf.reduce_sum(pos_weights*link_loss) / (tf.reduce_sum(pos_weights) + 1e-5) + tf.reduce_sum(neg_weights*link_loss) / (tf.reduce_sum(neg_weights) + 1e-5)
            L_loss = L_loss * tf.reduce_sum(maps[:, 0])

        return T_loss, L_loss, T_loss + L_loss

    def infer(self, input):
        with tf.variable_scope('vgg_base'):
            base = self.vgg_base(input)
        with tf.variable_scope('prediction'):
            prediction = self.prediction(base)
        return prediction

    def summary(self, input, maps, weights, prediction, T_loss, L_loss, loss):
        with tf.device("/device:cpu:0"):
            imgsummary = []
            imgsummary.append(tf.summary.image('inputimg', input[0]))
            for i in range(9):
                imgsummary.append(tf.summary.image('map_%d'%(i,), maps[0, :, :, i:i+1]))
                imgsummary.append(tf.summary.image('predict_%d'%(i,), tf.nn.softmax(prediction[0, :, :, 2*i:2*i+2])[:, :, :, 1:2]))

            losssummary = []
            losssummary.append(tf.summary.scalar('T_loss', T_loss))
            losssummary.append(tf.summary.scalar('L_loss', L_loss))
            losssummary.append(tf.summary.scalar('loss', loss))

            return tf.summary.merge(imgsummary), tf.summary.merge(losssummary)

    def loss(self, input, maps, weights, withsum=False):
        prediction = self.infer(input)
        with tf.variable_scope('loss'):
            T_loss, L_loss, loss = self.build_loss(prediction, maps, weights)
        if withsum:
            return loss, self.summary(input, maps, weights, prediction, T_loss, L_loss, loss)
        else:
            return loss, None

    def __init__(self, params):
        self.parameters = params
        self._scope = 'PixelLinkNetwork'

    def get_training_func(self, initializer):
        def training_fn(features, params=None, reuse=None):
            if params is None:
                params = self.parameters
            with tf.variable_scope(self._scope, initializer=initializer,
                                   reuse=reuse, custom_getter=cpu_variable_getter):
                loss, summary = self.loss(features['input_img'], features['Labels'][:, :, :, 0:9], features['Labels'][:, :, :, 9], not bool(reuse))
                return loss
        return training_fn

    def get_evaluation_func(self):
        def evaluation_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)
            # params.dropout = 0.0
            # params.use_variational_dropout = False
            # params.label_smoothing = 0.0

            with tf.variable_scope(self._scope):
                prediction = model_graph(features, "eval", params)

            return prediction

        return evaluation_fn

    def get_inference_func(self):
        def inference_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)
            params.dropout = 0.0
            params.use_variational_dropout = False
            params.label_smoothing = 0.0

            with tf.variable_scope(self._scope):
                logits = model_graph(features, "infer", params)

            return logits

        return inference_fn

