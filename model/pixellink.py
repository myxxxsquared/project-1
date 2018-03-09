
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
        with tf.variable_scope(name):
            conv = tf.nn.conv2d(
                input,
                tf.get_variable(
                    'kernel', dtype=tf.float32, shape=shape, initializer=xavier(), trainable=True),
                strides=(1, 1, 1, 1),
                padding="SAME",
                name='conv')
            conv = tf.nn.bias_add(
                conv,
                tf.get_variable(
                    'bias', shape=(shape[3], ), trainable=True, initializer=tf.zeros_initializer()),
                name='biasadd'
            )
            return conv

    def pool(self, input, stride, name):
        return tf.nn.max_pool(
            input,
            ksize=(1, 2, 2, 1),
            strides=stride,
            padding="SAME",
            name=name)

    def pre_processing(self, image):
        with tf.variable_scope("pre_processing"):
            return (image - np.reshape((123.68, 116.78, 103.94), (1, 1, 1, 3))) / 256

    def vgg_base(self, input_image):
        # layer = [
        #     'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',  # part1
        #     'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',  # part2
        #     'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        #     'relu3_3', 'pool3',                                   # part3
        #     'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        #     'relu4_3', 'pool4',                                   # part4
        #     'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        #     'relu5_3', 'pool5',                                   # part5
        #     'conv6_1', 'relu6_1',  'conv6_2', 'relu6_2'           # part6
        # ]

        layer = [
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',  # part1
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',  # part2
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
            'relu3_3', 'pool3',                                   # part3
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
            'relu4_3', 'pool4',                                   # part4
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
            'relu5_3', 'pool5',                                   # part5
            'conv6_1', 'relu6_1',                                 # part6
            'conv7_1', 'relu7_1'                                  # part7
        ]

        # filter_shapes = {
        #     # part1
        #     'conv1_1': (3, 3, 3, 64), 'conv1_2': (3, 3, 64, 64),
        #     # part2
        #     'conv2_1': (3, 3, 64, 128), 'conv2_2': (3, 3, 128, 128),
        #     # part3
        #     'conv3_1': (3, 3, 128, 256), 'conv3_2': (3, 3, 256, 256), 'conv3_3': (3, 3, 256, 256),
        #     # part4
        #     'conv4_1': (3, 3, 256, 512), 'conv4_2': (3, 3, 512, 512), 'conv4_3': (3, 3, 512, 512),
        #     # part4
        #     'conv5_1': (3, 3, 512, 512), 'conv5_2': (3, 3, 512, 512), 'conv5_3': (3, 3, 512, 512),
        #     # part 6
        #     'conv6_1': (3, 3, 512, 512), 'conv6_2': (3, 3, 512, 512)
        # }

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
            'conv6_1': (3, 3, 512, 1024),
            # part 7
            'conv7_1': (3, 3, 1024, 1024)
        }

        # pool_strides = {
        #     'pool1': (1, 2, 2, 1),
        #     'pool2': (1, 2, 2, 1),
        #     'pool3': (1, 2, 2, 1),
        #     'pool4': (1, 2, 2, 1),
        #     'pool5': (1, 1, 1, 1),
        # }

        pool_strides = {
            'pool1': (1, 2, 2, 1),
            'pool2': (1, 2, 2, 1),
            'pool3': (1, 2, 2, 1),
            'pool4': (1, 2, 2, 1),
            'pool5': (1, 3, 3, 1),
        }

        net = {}

        activations = self.pre_processing(input_image)
        for name in layer:
            Layer_type = name[:4]
            if Layer_type == 'conv':
                activations = self.conv2d(
                    activations, filter_shapes[name], name)
            elif Layer_type == 'relu':
                activations = tf.nn.relu(activations, name=name)
            elif Layer_type == 'pool':
                activations = self.pool(activations, pool_strides[name], name)
            net[name] = activations

        return net

    def prediction(self, vgg):
        # maps = [vgg[x] for x in
        #         {
        #             2: ['relu2_2', 'relu3_3', 'relu4_3', 'relu5_3', 'relu6_2'],
        #             4: ['relu3_3', 'relu4_3', 'relu5_3', 'relu6_2']}
        #         [self.parameters.output_scalar][::-1]]
        maps = [vgg[x] for x in
                {
                    2: ['relu2_2', 'relu3_3', 'relu4_3', 'relu5_3', 'conv7_1'],
                    4: ['relu3_3', 'relu4_3', 'relu5_3', 'relu6_2']}
                [self.parameters.output_scalar][::-1]]

        with tf.variable_scope('T'):
            T = self.prediction_block(maps, 2)
            print('T', T)
        with tf.variable_scope('L'):
            L = self.prediction_block(maps, 16)
            print('L', L)
        return tf.concat([T, L], axis=3)

    def prediction_block(self, maps, ochannels):
        # print('-----')
        # print(maps)
        # print('------')
        prediction = self.conv2d(
            maps[0], (1, 1, maps[0].shape[3], ochannels), 'conv_0')
        prediction = tf.nn.relu(prediction)
        for i in range(1, len(maps)):
            ff = maps[i]
            ffsize = (ff.shape[1].value, ff.shape[2].value)
            if not ffsize[0] or not ffsize[1]:
                dynamic_shape = tf.shape(ff)
                ffsize = tf.convert_to_tensor(
                    (dynamic_shape[1], dynamic_shape[2]))
            prediction = tf.image.resize_images(prediction, ffsize)  \
                + self.conv2d(maps[i], (1, 1, maps[i].shape[3],
                                        ochannels), 'conv_%d' % (i,))
        return tf.nn.relu(self.conv2d(prediction, (1, 1, ochannels, ochannels), 'conv_o'))

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
        weights = tf.reshape(weights, (-1, 1))

        cross_entropy = []
        for i in range(9):
            cross_entropy.append(tf.expand_dims(tf.losses.softmax_cross_entropy(
                tf.concat([1 - maps[:, i:i+1], maps[:, i:i+1]], axis=1), prediction[:, 2*i:2*i+2], reduction=tf.losses.Reduction.NONE), 1))

        with tf.name_scope('T'):
            pos_region = maps[:, 0:1]
            neg_region = 1 - pos_region
            posnum = tf.reduce_sum(pos_region) + 1e-5
            negsum = tf.reduce_sum(neg_region) + 1e-5
            k = tf.cast(tf.reduce_min(
                (r*posnum + 1, negsum)), tf.int32)

            # weighted_loss = cross_entropy[0] * weights
            weighted_loss = cross_entropy[0]
            pos_loss = pos_region * weighted_loss
            neg_loss = neg_region * cross_entropy[0]
            pos_loss, neg_loss = tf.squeeze(pos_loss, 1), tf.squeeze(neg_loss, 1)
            # print(neg_loss.dtype)
            # print(tf.nn.top_k(neg_loss, k=k).shape)
            # print(tf.nn.top_k(neg_loss, k=k).dtype)
            # T_loss = (tf.reduce_sum(pos_loss) +
            #           tf.reduce_mean(tf.nn.top_k(neg_loss, k=k).values)*r) / (1 + r)
            T_loss = (tf.reduce_mean(pos_loss) +
                      tf.reduce_mean(tf.nn.top_k(neg_loss, k=k).values))
            T_loss = lambda_ * T_loss

        with tf.name_scope('L'):
            pos_region = maps[:, 1:9]
            neg_region = 1 - pos_region
            link_loss = tf.concat(cross_entropy[1:9], 1)
            weights = tf.reshape(weights, (-1, 1))
            pos_weights = pos_region * weights
            neg_weights = neg_region * weights
            L_loss = tf.reduce_sum(pos_weights*link_loss) / (tf.reduce_sum(pos_weights) + 1e-5) + \
                tf.reduce_sum(neg_weights*link_loss) / \
                (tf.reduce_sum(neg_weights) + 1e-5)
            L_loss = L_loss


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
            imgsummary.append(tf.summary.image('inputimg', input[0:1]))
            imgsummary.append(tf.summary.image('weight', tf.expand_dims(weights[0:1], 3)))
            for i in range(9):
                imgsummary.append(tf.summary.image('map_%d' %
                                                   (i,), maps[0:1, :, :, i:i+1]))
                imgsummary.append(tf.summary.image('predict_%d' % (i,), tf.nn.softmax(
                    prediction[0:1, :, :, 2*i:2*i+2])[:, :, :, 1:2]))

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

    def __init__(self, params, name='PixelLinkNetwork'):
        self.parameters = params
        self._scope = name

    def get_training_func(self, initializer):
        def training_fn(features, params=None, reuse=None):
            if params is None:
                params = self.parameters
            with tf.variable_scope(self._scope, initializer=initializer,
                                   reuse=reuse):
                loss, summary = self.loss(
                    features['input_img'], features['Labels'][:, :, :, 0:9], features['Labels'][:, :, :, 9], not bool(reuse))
                returnval = loss, summary
                return returnval
        return training_fn

    def get_evaluation_func(self):
        def evaluation_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                prediction = self.infer(features['input_img'])

            return prediction

        return evaluation_fn

    def get_inference_func(self):
        def inference_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                input_img = features['input_img']
                prediction = self.infer(input_img)
                lens = features['lens']
                cnts = features['cnts']
                care = features['care']

            return {'prediction':prediction,
                    'input_img': input_img,
                    'lens':lens,
                    'cnts':cnts,
                    'care':care}

        return inference_fn
