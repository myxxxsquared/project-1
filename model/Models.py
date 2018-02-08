from . import Basenet
import tensorflow as tf
import os, time
import numpy as np
from .yellowfin import *

from tensorflow.contrib.layers import xavier_initializer_conv2d as xavier

def model_graph(features, params):
    img = features['img']
    TR = features['TR']
    TCL = features['TCL']
    radius = features['radius']
    cos_theta = features['cos_theta']
    sin_theta = features['sin_theta']
    alpha = tf.Variable(0.5, dtype=tf.float32)
    loss = tf.reduce_sum(img)*alpha+tf.reduce_sum(img)*(1-alpha)
    return loss

class  model(object):
    def __init__(self, params):
        self.params = params
        self._scope = 'model'

    def get_training_func(self, initializer):
        def training_fn(features, params=None):
            if params is None:
                params = self.params
            with tf.variable_scope(self._scope, initializer=initializer,
                                   reuse=tf.AUTO_REUSE):
                loss = model_graph(features, params)
                return loss

        return training_fn


    # this function is to build the network stem from BaseNet
#     def _build_network(self, features):
#         img, TR, TCL, radius, cos_theta, sin_theta = features
#         img = tf.cast(img, tf.float32)
#         TR = tf.cast(TR, tf.float32)
#         TCL = tf.cast(TCL, tf.float32)
#         radius = tf.cast(radius, tf.float32)
#         cos_theta = tf.cast(cos_theta, tf.float32)
#         sin_theta = tf.cast(sin_theta, tf.float32)
#
#         # self.input_image = tf.placeholder(tf.float32,
#         #                                   shape=(None,
#         #                                          *(self.configs.image_size)),
#         #                                   name="InputImage")
#         # self.Labels = tf.placeholder(tf.float32,
#         #                              shape=(None, self.configs.Label_size[0], self.configs.Label_size[1],
#         #                                     self.configs.Label_size[2]),
#         #                              name="Labels")
#         self.input_image = img
#         self.Labels = np.stack((TR, TCL, radius, cos_theta))
#
#
#         basenets = {'vgg16': Basenet.VGG16, 'vgg19': Basenet.VGG16,
#                     'resnet': Basenet.ResNet}  # for resnet :  'resnet-layer_number'
#
#         self.basenet = basenets[self.configs.basenet](self.configs, self.logs)
#         self.basenet.net_loading(input_image=self.input_image,
#                                  layer=self.configs.basenet[self.configs.basenet.find('-') + 1:])
#         # self.logs['info']('Network built: backbone.')
#         self._add_prediction_block()
#         # self.logs['info']('Network built: prediction layer.')
#         self._build_loss()
#         # self.logs['info']('Network built: loss function.')
#         # self.logs['info']('Network built: train step.')
#         # self.logs['info']('Network built: all done.')
#         self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
#         return self.total_loss
#
#     # this function is to build blocks for predictions of pixels
#     def _add_prediction_block(self):
#         # prediciton stage
#         params = self.configs.US_Params.split(' ')
#         activation_functions = {'None': None, 'ReLU': tf.nn.relu, 'cReLU': tf.nn.crelu, 'softmax': tf.nn.softmax}
#         if self.configs.upsampling == 'DeConv':
#             kwargs = {'output_channel': None,
#                       'kernel_size': (int(params[0]), int(params[1])),
#                       'strides': (int(params[2]), int(params[3])),
#                       'padding': params[4],
#                       'activation': activation_functions[params[5]],
#                       'name': None}
#         else:
#             kwargs = {
#                 'size': int(params)
#             }
#         self.basenet.pipe['prediction'] = {}
#         # stage 1
#         self.basenet.pipe['prediction']['stage1'] = h = self.basenet.pipe['output_pipe']['stage5']
#
#         # stage 2-4
#         for stage in range(2, self.configs.Predict_stage + 1):  # 4
#             kwargs['output_channel'] = h.shape.as_list()[-1] // 2
#             kwargs['name'] = 'stage%d_US' % stage
#             h = self.basenet._upsampling(h, upsampling=self.configs.upsampling, **kwargs)
#             h = tf.concat([h, self.basenet.pipe['output_pipe']['stage' + str(6 - stage)]], axis=-1)
#             h = self.basenet._ConvLayer(h,
#                                         shape=(
#                                             1, 1, h.shape.as_list()[-1],
#                                             self.configs.predict_channels[stage - 2]),
#                                         padding=self.configs.padding, strides=(1, 1, 1, 1),
#                                         name='Predict_stage_' + str(stage) + '_Conv_1')
#             h = tf.nn.relu(h)
#             h = self.basenet._ConvLayer(h,
#                                         shape=(
#                                             3, 3, h.shape.as_list()[-1],
#                                             self.configs.predict_channels[stage - 2]),
#                                         padding=self.configs.padding, strides=(1, 1, 1, 1),
#                                         name='Predict_stage_' + str(stage) + '_Conv_2')
#             self.basenet.pipe['prediction']['stage' + str(stage)] = h = tf.nn.relu(h)
#
#         h = self.basenet._ConvLayer(h,
#                                     shape=(
#                                         3, 3, h.shape.as_list()[-1],
#                                         self.configs.predict_channels[self.configs.Predict_stage - 1]),
#                                     padding=self.configs.padding, strides=(1, 1, 1, 1),
#                                     name='Predict_stage_penul')
#         h = tf.nn.relu(h)
#
#         self.prediction = h = tf.image.resize_images(self.basenet._ConvLayer(h,
#                                                                              shape=(
#                                                                                  3, 3, h.shape.as_list()[-1],
#                                                                                  self.configs.Label_size[2] + 1),
#                                                                              padding=self.configs.padding,
#                                                                              strides=(1, 1, 1, 1),
#                                                                              name='Predict_stage_final'),
#                                                      size=(self.configs.Label_size[0], self.configs.Label_size[1]))
#         # tf.image.resize_images(inputs,size=(kwargs['size'],kwargs['size']))
#
#         ##TODO at present, the score map is predicted in two channels (1 for pos, 1 for neg)
#
#     # this function is to build loss function
#     def _build_loss(self):
#         # model output is : self.prediction
#         # ground truth is : self.Labels
#         # here the model only consider a single score map: text center line or not
#         # TODO: taking multiple score maps: e.g. text center line vs. text region vs. background
#         def pos_mask():
#             # from self.Labels[:,:,:,0:1]
#             return self.Labels[:, :, :, 0:1] > 0
#
#         pos = tf.cast(pos_mask(), tf.float32)
#         pos_num = tf.reduce_sum(pos)
#         neg_num = tf.cast(3 * pos_num, tf.int32)  # in case, OHNM is used
#         # score map loss
#         singel_labels = flatten(self.Labels[:, :, :, 0:1])
#         one_hot_labels = tf.concat([1 - singel_labels, singel_labels], axis=-1)
#         loss = tf.losses.softmax_cross_entropy(one_hot_labels,
#                                                flatten(self.prediction[:, :, :, 0:2]),
#                                                reduction=tf.losses.Reduction.NONE)
#
#         def pos_loss():
#             pos_flatten = tf.reshape(flatten(pos), shape=(-1,))
#             pos_loss = loss * pos_flatten
#             neg_losses = loss * (1 - pos_flatten)
#             neg_loss = tf.nn.top_k(neg_losses, k=neg_num).values
#             return (tf.reduce_sum(pos_loss) +
#                     tf.reduce_sum(neg_loss)) / pos_num
#
#         def neg_loss():
#             return tf.constant(0.0, dtype=tf.float32)
#
#         self.score_loss = tf.cond(pos_num > 0, pos_loss, neg_loss)
#
#         # geo loss
#         geo_attr = ['geo_1', 'geo_2', 'geo_3']
#         self.geo_loss = []
#         self.total_loss = self.score_loss
#         tf.summary.scalar("score_map_Loss", self.score_loss)
#         for i in range(self.configs.Label_size[2] - 2):
#             self.geo_loss.append(smooth_l1_loss(flatten(self.Labels[:, :, :, i + 1:i + 2] * pos),
#                                                 flatten(self.prediction[:, :, :, i + 2:i + 3] * pos)
#                                                 ) / pos_num)
#             self.total_loss += self.geo_loss[-1]
#             tf.summary.scalar('%s_loss' % geo_attr[i], self.geo_loss[-1])
#         self.summary_op = tf.summary.merge_all()
#         tf.summary.scalar("Train_Loss", self.total_loss)
#         i1 = tf.summary.image('original_image', self.input_image[0:1, :, :, :], 1)  # printing only the first pic
#         i2 = tf.summary.image('TCL_GT_score_map', self.Labels[0:1, :, :, 0:1], 1)  # printing only the first pic
#         i3 = tf.summary.image('TR_GT_score_map', self.Labels[0:1, :, :, 4:5], 1)
#         tr = tf.cast(self.prediction[0:1, :, :, 6:7] > self.prediction[0:1, :, :, 5:6], tf.uint8)
#         i4 = tf.summary.image('TCL_predicted_score_map',
#                               tr * tf.cast(self.prediction[0:1, :, :, 1:2] > self.prediction[0:1, :, :, 0:1],
#                                            tf.uint8) * 255, 1)
#         i5 = tf.summary.image('TR_predicted_score_map',
#                               tr, 1)
#         self.image_summary_op = tf.summary.merge([i1, i2, i3, i4, i5])
#
#     def get_training_func(self, initializer):
#         def fn(features):
#             return self._build_network(features)
#         return fn
#
#
#
# def flatten(output):
#     """
#     :param output: Dict(layer=tensor[batch,h,w,2]) for score map only
#     :return: tensor[batch,C]
#     """
#     shape = output.shape.as_list()
#     if shape[-1] == 2:
#         return tf.reshape(output, shape=(-1, 2))
#     else:
#         return tf.reshape(output, shape=(-1, 1))
#
#
# def smooth_l1_loss(pred, target):
#     diff = pred - target
#     abs_diff = tf.abs(diff)
#     abs_diff_lt_1 = tf.less(abs_diff, 1)
#     loss = tf.where(abs_diff_lt_1, 0.5 * tf.square(abs_diff), abs_diff - 0.5)
#     return tf.reduce_sum(loss)