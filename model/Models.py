import tensorflow as tf
from . import Basenet
import os, time
from .yellowfin import *


class model():
    def __init__(self, params, configs):
        self.params = params
        self.configs = configs

    def _add_prediction_block(self):
        #prediciton stage
        params=self.configs.US_Params.split(' ')
        activation_functions={'None':None,'ReLU':tf.nn.relu,'cReLU':tf.nn.crelu,'softmax':tf.nn.softmax}
        if self.configs.upsampling == 'DeConv':
            kwargs = {'output_channel': None,
                      'kernel_size': (int(params[0]),int(params[1])),
                      'strides': (int(params[2]),int(params[3])),
                      'padding': params[4],
                      'activation': activation_functions[params[5]],
                      'name': None}
        else:
            kwargs={
                'size':int(params)
            }
        self.basenet.pipe['prediction']={}
        #stage 1
        self.basenet.pipe['prediction']['stage1']= h = self.basenet.pipe['output_pipe']['stage5']

        #stage 2-4
        for stage in range(2,self.configs.Predict_stage+1):#4
            kwargs['output_channel']=h.shape.as_list()[-1]//2
            kwargs['name']='stage%d_US'%stage
            h=self.basenet._upsampling(h,upsampling=self.configs.upsampling,**kwargs)
            h=tf.concat([h,self.basenet.pipe['output_pipe']['stage'+str(6-stage)]],axis=-1)
            h=self.basenet._ConvLayer(h,
                                      shape=(
                                             1, 1, h.shape.as_list()[-1],
                                             self.configs.predict_channels[stage - 2]),
                                      padding=self.configs.padding, strides=(1,1,1,1),
                                      name='Predict_stage_'+str(stage)+'_Conv_1')
            h=tf.nn.relu(h)
            h = self.basenet._ConvLayer(h,
                                        shape=(
                                            3, 3, h.shape.as_list()[-1],
                                            self.configs.predict_channels[stage - 2]),
                                        padding=self.configs.padding, strides=(1, 1, 1, 1),
                                        name='Predict_stage_' + str(stage)+'_Conv_2')
            self.basenet.pipe['prediction']['stage'+str(stage)] = h = tf.nn.relu(h)

        h = self.basenet._ConvLayer(h,
                                    shape=(
                                        3, 3, h.shape.as_list()[-1],
                                        self.configs.predict_channels[self.configs.Predict_stage-1]),
                                    padding=self.configs.padding, strides=(1, 1, 1, 1),
                                    name='Predict_stage_penul')
        h = tf.nn.relu(h)

        self.prediction= h = self.basenet._ConvLayer(h,
                                    shape=(
                                        3, 3, h.shape.as_list()[-1],
                                        self.configs.Label_size[2]+1),
                                    padding=self.configs.padding, strides=(1, 1, 1, 1),
                                    name='Predict_stage_final')

    def _build_loss(self, Labels):
        # TODO: taking multiple score maps: e.g. text center line vs. text region vs. background
        ##TR score loss

        def pos_mask():
            # from self.Labels[:,:,:,0:1]
            return Labels[:, :, :, 0:1] > 0

        pos = tf.cast(pos_mask(), tf.float32)

        ##TCL score loss

        def pos_mask():
            # from self.Labels[:,:,:,0:1]
            return Labels[:, :, :, 0:1] > 0

        pos = tf.cast(pos_mask(), tf.float32)

        ##TCL score loss
        with tf.name_scope('tcl_score_loss'):
            pos_num = tf.reduce_sum(pos)
            neg_num = tf.cast(3 * pos_num, tf.int32)  # in case, OHNM is used
            # score map loss
            singel_labels = flatten(Labels[:, :, :, 0:1])
            one_hot_labels = tf.concat([1 - singel_labels, singel_labels], axis=-1)
            loss = tf.losses.softmax_cross_entropy(one_hot_labels,
                                                   flatten(self.prediction[:, :, :, 0:2]),
                                                   reduction=tf.losses.Reduction.NONE)

            def pos_loss():
                pos_flatten = tf.reshape(flatten(pos), shape=(-1,))
                pos_loss = loss * pos_flatten  # *pos_flatten_TR
                neg_loss = loss * (1 - pos_flatten)  # *pos_flatten_TR
                # neg_loss=tf.nn.top_k(neg_losses,k=neg_num).values
                return (tf.reduce_sum(pos_loss) / pos_num +
                        tf.reduce_sum(neg_loss) / pos_num)

            def neg_loss():
                return tf.constant(0.0, dtype=tf.float32)

            score_loss = tf.cond(pos_num > 0, pos_loss, neg_loss)
            print('!!!!!!!!! score loss: !!!!!!', score_loss)

            tf.add_to_collection('my_losses', score_loss)

        # total_loss = score_loss + TR_score_loss #for training
        # geo loss
        geo_attr = ['theta', 'curvature', 'thickness']
        geo_weights = [1, 1, 1]
        for i in range(len(geo_attr)):
            loss_name = '{}_loss'.format(geo_attr[i])
            with tf.name_scope(loss_name):
                print('!!!!!!!!!!! Labels; !!!!!!!!!!!!', Labels)
                print('!!!!!!!!!!! prediction: !!!!!!!!!!!!', self.prediction)
                print('!!!!!!!!!!! pos !!!!!!!!!!!!', pos)

                cur_loss = geo_weights[i] * smooth_l1_loss(flatten(Labels[:, :, :, i + 1:i + 2] * pos),
                                                           flatten(self.prediction[:, :, :,
                                                                   i + 2:i + 3] * pos)) / pos_num
                print('!!!!!!!!!!! cur_loss: !!!!!!!!!', cur_loss)
                tf.add_to_collection('my_losses', cur_loss)


    def get_loss(self, features):

        img, TR, TCL, radius, cos_theta, sin_theta = features
        img = tf.reshape(img, (512,512,3))
        TR = tf.reshape(TR, (512,512))
        TCL = tf.reshape(TCL, (512,512))
        radius = tf.reshape(radius, (512,512))
        cos_theta = tf.reshape(cos_theta, (512,512))
        sin_theta = tf.reshape(sin_theta, (512,512))
        Labels = tf.stack((TR, TCL, radius, cos_theta, sin_theta))
        basenets={'vgg16':Basenet.VGG16,'vgg19':Basenet.VGG16,'resnet':Basenet.ResNet}#for resnet :  'resnet-layer_number'

        self.basenet = basenets[self.params.basenet](self.params,None)
        self.basenet.net_loading(input_image=img, layer=self.params.basenet[self.params.basenet.find('-')+1:])

        self._add_prediction_block()
        self._build_loss(Labels)
        losses = tf.get_collection('my_losses')

        total_clone_loss = tf.add_n(losses, name='total_clone_loss')
        return total_clone_loss

    def get_training_func(self, initializer):
        return self.get_loss



def flatten(output):
    """
    :param output: Dict(layer=tensor[batch,h,w,2]) for score map only
    :return: tensor[batch,C]
    """
    shape = output.shape.as_list()
    if shape[-1] == 2:
        return tf.reshape(output, shape=(-1, 2))
    else:
        return tf.reshape(output, shape=(-1, 1))


def smooth_l1_loss(pred, target):
    diff = pred - target
    abs_diff = tf.abs(diff)
    abs_diff_lt_1 = tf.less(abs_diff, 1)
    loss = tf.where(abs_diff_lt_1, 0.5 * tf.square(abs_diff), abs_diff - 0.5)
    return tf.reduce_sum(loss)