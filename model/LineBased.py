import tensorflow as tf
import model.Basenet as Basenet
import copy
from tensorflow.contrib.layers import xavier_initializer_conv2d as xavier


def _flatten(output):
    """
    :param output: Dict(layer=tensor[batch,h,w,2]) for score map only
    :return: tensor[batch,C]
    """
    shape = output.shape.as_list()
    if shape[-1] == 2:
        return tf.reshape(output, shape=(-1, 2))
    else:
        return tf.reshape(output, shape=(-1, 1))


def _smooth_l1_loss(pred, target):
    diff = pred - target
    abs_diff = tf.abs(diff)
    abs_diff_lt_1 = tf.less(abs_diff, 1)
    loss = tf.where(abs_diff_lt_1, 0.5 * tf.square(abs_diff), abs_diff - 0.5)
    return tf.reduce_sum(loss)


def _ConvLayer(inputs, filters=None, bias=None, shape=None, strides=(1, 1, 1, 1), padding="SAME", name="Conv",
               trainable=True, initializer=xavier):
    """
    :param inputs:
    :param filters:
    :param bias:
    :param shape: (k1,k2,input_channel,output_channel)
    :param padding:"SAME"(zero),"VALID"(no)
    :param name:
    :param trainable:
    :return:
    """
    if filters:
        conv = tf.nn.conv2d(
            inputs,
            tf.get_variable(name, initializer=filters, dtype=tf.float32, trainable=trainable),
            strides=strides,
            padding=padding,
            name=name)
    elif shape:
        conv = tf.nn.conv2d(
            inputs,
            tf.get_variable(name, dtype=tf.float32, shape=shape, initializer=initializer(), trainable=True),
            strides=strides,
            padding=padding,
            name=name)
    else:
        raise ValueError('At least one of the following should be passed: shape, filters')
    if bias:
        return tf.nn.bias_add(conv, bias)
    else:
        return tf.nn.bias_add(conv, tf.get_variable(name + '_bias', dtype=tf.float32, shape=(shape[-1]),
                                                    initializer=initializer()))


def _upsampling(inputs, upsampling='DeConv', **kwargs):
    """
    :param inputs:
    :param upsampling: UpSam algorithm, 'BiLi'or'DeConv' or 'unpooling'(?)
    :param name: str
    :param kwargs: dict{}
    :return:
    conv2d_transpose(
        inputs,
        filters,
        kernel_size,
        strides=(1, 1),
        padding='valid',
        data_format='channels_last',
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        reuse=None
    )
    """
    if upsampling == 'DeConv':
        return tf.layers.conv2d_transpose(inputs=inputs,
                                          filters=kwargs['output_channel'],
                                          kernel_size=kwargs['kernel_size'],
                                          strides=kwargs['strides'],
                                          padding=kwargs['padding'],
                                          activation=kwargs['activation'],
                                          name=kwargs['name'])  # '3 3 2 2 same ReLU'
    if upsampling == 'BiLi':
        return tf.image.resize_images(inputs, size=(kwargs['size'], kwargs['size']))  # '3'

    raise ValueError('No such UpSampling methods.')


def _build_back_bone(params, input_image):
    basenets = {'vgg16': Basenet.VGG16, 'vgg19': Basenet.VGG16,
                'resnet': Basenet.ResNet}  # for resnet :  'resnet-layer_number'
    basenet = basenets[params.basenet](params)
    basenet.net_loading(input_image=input_image) #TODO: layer args
    return basenet.pipe


def _add_prediction_block(params, pipe):
    # prediciton stage
    US_params = params.US_Params.split(' ')
    activation_functions = {'None': None, 'ReLU': tf.nn.relu, 'cReLU': tf.nn.crelu, 'softmax': tf.nn.softmax}
    if params.upsampling == 'DeConv':
        kwargs = {'output_channel': None,
                  'kernel_size': (int(US_params[0]), int(US_params[1])),
                  'strides': (int(US_params[2]), int(US_params[3])),
                  'padding': US_params[4],
                  'activation': activation_functions[US_params[5]],
                  'name': None}
    else:
        kwargs = {
            'size': int(US_params)
        }
    pipe['prediction'] = {}
    # stage 1
    pipe['prediction']['stage1'] = h = pipe['output_pipe']['stage5']

    # stage 2-4
    for stage in range(2, params.Predict_stage + 1):  # [2,5)
        kwargs['output_channel'] = h.shape.as_list()[-1] // 2
        kwargs['name'] = 'stage%d_US' % stage
        h = _upsampling(h, upsampling=params.upsampling, **kwargs)
        h = tf.concat([h, pipe['output_pipe']['stage' + str(6 - stage)]], axis=-1)
        h = _ConvLayer(h,
                       shape=(
                            1, 1, h.shape.as_list()[-1],
                            params.predict_channels[stage - 2]),
                       padding=params.padding, strides=(1, 1, 1, 1),
                       name='Predict_stage_' + str(stage) + '_Conv_1')
        h = tf.nn.relu(h)
        h = _ConvLayer(h,
                       shape=(
                            3, 3, h.shape.as_list()[-1],
                            params.predict_channels[stage - 2]),
                       padding=params.padding, strides=(1, 1, 1, 1),
                       name='Predict_stage_' + str(stage) + '_Conv_2')

        pipe['prediction']['stage' + str(stage)] = h = tf.nn.relu(h)

    # stage 5-6
    kwargs['output_channel'] = params.predict_channels[-1]
    kwargs['name'] = 'stage%d_US' % (-2)
    h = _upsampling(h, upsampling=params.upsampling, **kwargs)
    kwargs['name'] = 'stage%d_US' % (-1)
    h = _upsampling(h, upsampling=params.upsampling, **kwargs)
    h = _ConvLayer(h,
                   shape=(
                        3, 3, h.shape.as_list()[-1],
                        params.predict_channels[params.Predict_stage - 1]),
                   padding=params.padding, strides=(1, 1, 1, 1),
                   name='Predict_stage_penul')
    h = tf.nn.relu(h)

    # final stage to make prediction
    prediction = h = _ConvLayer(h,
                                shape=(
                                    3, 3, h.shape.as_list()[-1],  # TODO kernel size should be (1,1) here(?)
                                    params.Label_size[2] + 1 + 1),
                                # plus one for text region map
                                padding=params.padding, strides=(1, 1, 1, 1),
                                name='Predict_stage_final')  # ,

    # regularize cos and sin to a squared sum of 1
    cos_sin_sum = tf.sqrt(prediction[:, :, :, 3] * prediction[:, :, :, 3] + prediction[:, :, :, 4] * prediction[:, :, :, 4])
    cos = prediction[:, :, :, 3] / cos_sin_sum
    sin = prediction[:, :, :, 4] / cos_sin_sum
    return tf.stack([prediction[:, :, :, 0], prediction[:, :, :, 1], prediction[:, :, :, 2], cos, sin, prediction[:, :, :, 5], prediction[:, :, :, 6]], axis=-1)


def _build_loss(Labels, prediction):
    with tf.name_scope('TR_loss'):
        def pos_mask_TR():
            # from self.Labels[:,:,:,0:1]
            return Labels[:, :, :, 4:5] > 0
        pos_TR = tf.cast(pos_mask_TR(), tf.float32)
        pos_num_TR = tf.reduce_sum(pos_TR) + 1e-3
        neg_num_TR = tf.cast(3 * pos_num_TR + 1, tf.int32)  # in case, OHNM is used
        # TR score map loss
        singel_labels_TR = _flatten(Labels[:, :, :, 4:5])
        one_hot_labels_TR = tf.concat([1 - singel_labels_TR, singel_labels_TR], axis=-1)
        loss_TR = tf.losses.softmax_cross_entropy(one_hot_labels_TR,
                                                  _flatten(prediction[:, :, :, 5:7]),
                                                  reduction=tf.losses.Reduction.NONE)
        pos_flatten_TR = tf.reshape(_flatten(pos_TR), shape=(-1,))

        pos_loss_TR = loss_TR * pos_flatten_TR
        neg_losses_TR = loss_TR * (1 - pos_flatten_TR)
        neg_loss_TR = tf.nn.top_k(neg_losses_TR, k=tf.reduce_min((neg_num_TR, tf.size(neg_losses_TR)))).values
        TR_score_loss = (tf.reduce_sum(pos_loss_TR) / pos_num_TR +
                         tf.reduce_sum(neg_loss_TR) / pos_num_TR) # top_k in use

    with tf.name_scope('TCL_loss'):
        def pos_mask():
            # from self.Labels[:,:,:,0:1]
            return Labels[:, :, :, 0:1] > 0
        pos = tf.cast(pos_mask(), tf.float32)
        pos_num = tf.reduce_sum(pos) + 1e-3
        neg_num = tf.cast(3 * pos_num + 1, tf.int32)  # in case, OHNM is used
        # score map loss
        singel_labels = _flatten(Labels[:, :, :, 0:1])
        one_hot_labels = tf.concat([1 - singel_labels, singel_labels], axis=-1)
        loss = tf.losses.softmax_cross_entropy(one_hot_labels,
                                               _flatten(prediction[:, :, :, 0:2]),
                                               reduction=tf.losses.Reduction.NONE)

        pos_flatten = tf.reshape(_flatten(pos), shape=(-1,))
        pos_loss = loss * pos_flatten * pos_flatten_TR
        neg_loss = loss * (1 - pos_flatten) * pos_flatten_TR
        # neg_loss=tf.nn.top_k(neg_losses,k=neg_num).values
        score_loss = (tf.reduce_sum(pos_loss) / pos_num +
                      tf.reduce_sum(neg_loss) / pos_num)# top_k not in use TODO: rebalance the pos/neg number !

    with tf.name_scope('Geo_loss'):
        geo_attr = ['radius', 'cos', 'sin']
        geo_loss = []
        total_loss = score_loss + TR_score_loss  # for training
        for i in range(4 - 1):  # self.configs.Label_size[2]-1):
            geo_loss.append(_smooth_l1_loss(_flatten(Labels[:, :, :, i + 1:i + 2] * pos),
                                           _flatten(prediction[:, :, :, i + 2:i + 3] * pos)
                                           ) / pos_num)
            total_loss += geo_loss[-1]

    return total_loss, score_loss, geo_loss, geo_attr, TR_score_loss


def model_graph(features, mode, params):
    if mode == 'train':
        input_img = features['input_img']
        Labels = features['Labels']
        pipe = _build_back_bone(params, input_img)
        prediction = _add_prediction_block(params, pipe)
        total_loss, score_loss, geo_loss, geo_attr, TR_score_loss = _build_loss(Labels, prediction)
        loss = total_loss
    else:
        loss = 0.0
    return loss


class Model(object):
    def __init__(self, params):
        self.parameters = params
        self._scope = 'LineBasedModel'

    def get_training_func(self, initializer):
        def training_fn(features, params=None, reuse=None):
            if params is None:
                params = self.parameters
            with tf.variable_scope(self._scope, initializer=initializer,
                                   reuse=reuse):
                loss = model_graph(features, "train", params)
                return loss

        return training_fn

    def get_evaluation_func(self):
        def evaluation_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)
            params.dropout = 0.0
            params.use_variational_dropout = False
            params.label_smoothing = 0.0

            with tf.variable_scope(self._scope):
                score = model_graph(features, "eval", params)

            return score

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

