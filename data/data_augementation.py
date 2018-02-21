#   to_be_determined
#   please refer to https://github.com/aleju/imgaug
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
from random import shuffle, randint, choice, random
import copy


def _get_seq(self, affine=False):
    if affine:
        affine = [
            iaa.Affine(rotate=360*np.random.rand()-180),
            iaa.Affine(shear=80*np.sin(np.random.rand()*np.pi/2)-40),
            iaa.Affine(scale={"x": 0.4*np.random.rand()+0.8, "y": 0.4*np.random.rand()+0.8})
        ]
        return iaa.Sequential([choice(affine)])#

    add_augmentation_list = [
        iaa.Add((-30, 30), per_channel=True),
        iaa.Add((-30, 30), per_channel=False),
        iaa.AddElementwise((-30, 30), per_channel=False),
        iaa.AddElementwise((-30, 30), per_channel=True),
        iaa.Invert(p=0.2, per_channel=True),
        iaa.Invert(p=0.2, per_channel=False),
        iaa.AddToHueAndSaturation((0, 80), True),
        iaa.Multiply((0.8, 1.2), per_channel=True),
        iaa.Multiply((0.8, 1.2), per_channel=False),
        iaa.MultiplyElementwise((0.8, 1.2), per_channel=True),
        iaa.MultiplyElementwise((0.8, 1.2), per_channel=False)
     ]

    blur_augmentation_list = [
        iaa.GaussianBlur((2, 3)),
        iaa.AverageBlur((2, 3)),
        iaa.MedianBlur((3, 5)),
        iaa.BilateralBlur((2, 3))
    ]

    noise_augmentation_list = [
        iaa.AdditiveGaussianNoise(0, (5, 20), per_channel=True),
        iaa.AdditiveGaussianNoise(0, (5, 20), per_channel=False),
        iaa.Dropout((0.05, 0.15), per_channel=False),
        iaa.Dropout((0.05, 0.15), per_channel=True),
        iaa.CoarseDropout((0.05, 0.15), size_percent=(0.5, 0.7))
        # iaa.SaltAndPepper((0.05, 0.15), per_channel=True),
        # iaa.SaltAndPepper((0.05, 0.15), per_channel=False)
    ]

    other_augmentation_list = [
        iaa.Sharpen((0.9, 0.11), (0.8, 1.2)),
        iaa.Emboss((0.9, 0.11), (0.3, 1.6)),
        iaa.EdgeDetect((0, 0.4)),
        iaa.Grayscale((0, 1))
    ]

    shuffle(add_augmentation_list)
    add_augmentation_list = add_augmentation_list[:randint(0, 1)]
    shuffle(other_augmentation_list)
    other_augmentation_list = other_augmentation_list[:randint(0, 1)]
    shuffle(blur_augmentation_list)
    blur_augmentation_list = blur_augmentation_list[:randint(0, 1)]
    shuffle(noise_augmentation_list)
    noise_augmentation_list = noise_augmentation_list[:randint(0, 1)]

    final_list = add_augmentation_list + noise_augmentation_list + blur_augmentation_list + other_augmentation_list

    if len(final_list) == 0:
        final_list.append(iaa.Noop())

    return iaa.Sequential(final_list[:2])

def _key_points(image_shape, point_list):
    """
    feed cnt and return ia.KeypointsOnImage object
    :param point_list: np.array size=(n,1,2)
           image_shape
    :return:
    """
    keypoint_list = []
    for i in range(point_list.shape[0]):
        keypoint_list.append(ia.Keypoint(x=point_list[i, 0, 0], y=point_list[i, 0, 1]))
    return ia.KeypointsOnImage(keypoint_list,
                               shape=ia.quokka(size=image_shape[:2]))

def _resize(input_data):
    """
    resize image under 512P to 512P
    :param input_data:
    :return:
    return scaled img
    """
    rate = 1
    if input_data['img'].shape[0] > input_data['img'].shape[1]:
        if True:  # input_data['img'].shape[1] < 512:
            rate = 512/input_data['img'].shape[1]
            seq = iaa.Sequential([
                iaa.Scale({'height': "keep-aspect-ratio", 'width': 512}, 'cubic')
            ])
            input_data['img'] = seq.augment_image(input_data['img'])
    else:
        if True:  # input_data['img'].shape[0] < 512:
            rate = 512/input_data['img'].shape[0]
            seq = iaa.Sequential([
                iaa.Scale({'height': 512, 'width': "keep-aspect-ratio"}, 'cubic')
            ])
            input_data['img'] = seq.augment_image(input_data['img'])
    if input_data['is_text_cnts']:
        input_data['contour'] = [np.cast['int32'](contour * rate) for contour in input_data['contour']]
    else:
        input_data['contour'] = [[np.cast['int32'](contour * rate) for contour in contours] for contours in
                                 input_data['contour']]
    input_data['center_point'] = [(np.cast['int32'](point[0] * rate),
                                   np.cast['int32'](point[1] * rate)) for point in input_data['center_point']]
    return input_data

def _resize_to_32x(input_data):
    """
    resize image to multilpes of 32
    :param input_data:
    :return:
    return scaled img
    """
    rate = 1

    p=0

    while True:
        if (input_data['img'].shape[0]+p)%32==0: row=input_data['img'].shape[0]+p;break
        if (input_data['img'].shape[0] - p) % 32 == 0: row = input_data['img'].shape[0] - p;break
        p+=1
    p=0
    while True:
        if (input_data['img'].shape[1]+p)%32==0: col=input_data['img'].shape[1]+p;break
        if (input_data['img'].shape[1] - p) % 32 == 0: col = input_data['img'].shape[1] - p;break
        p+=1

    if input_data['is_text_cnts']:
        input_data['contour'] = [np.cast['int32'](np.stack([contour[:,:,0] * row/input_data['img'].shape[0],
                                                            contour[:,:,1] * col/input_data['img'].shape[1]],
                                                           axis=-1)) for contour in input_data['contour']]
    else:
        input_data['contour'] = [[np.cast['int32'](np.stack([contour[:,:,0] * row/input_data['img'].shape[0],
                                                            contour[:,:,1] * col/input_data['img'].shape[1]],
                                                           axis=-1))
                                  for contour in contours] for contours in input_data['contour']]
    input_data['center_point'] = [(np.cast['int32'](point[0] * row/input_data['img'].shape[0]),
                                   np.cast['int32'](point[1] * col/input_data['img'].shape[1]))
                                  for point in input_data['center_point']]

    seq = iaa.Sequential([
        iaa.Scale({'height': row, 'width': col}, 'cubic')
    ])
    input_data['img'] = seq.augment_image(input_data['img'])

    return input_data

def _resize_to_512(input_data):
    """
    resize image to 512*512
    :param input_data:
    :return:
    return scaled img
    """

    if input_data['is_text_cnts']:
        input_data['contour'] = [np.cast['int32'](np.stack([contour[:,:,0] * 512/input_data['img'].shape[0],
                                                            contour[:,:,1] * 512/input_data['img'].shape[1]],
                                                           axis=-1)) for contour in input_data['contour']]
    else:
        input_data['contour'] = [[np.cast['int32'](np.stack([contour[:,:,0] * 512/input_data['img'].shape[0],
                                                            contour[:,:,1] * 512/input_data['img'].shape[1]],
                                                           axis=-1))
                                  for contour in contours] for contours in input_data['contour']]
    input_data['center_point'] = [(np.cast['int32'](point[0] * 512/input_data['img'].shape[0]),
                                   np.cast['int32'](point[1] * 512/input_data['img'].shape[1]))
                                  for point in input_data['center_point']]

    seq = iaa.Sequential([
        iaa.Scale({'height': 512, 'width': 512}, 'cubic')
    ])
    input_data['img'] = seq.augment_image(input_data['img'])

    return input_data

def _pad(input_data):
    h = input_data['img'].shape[0]
    w = input_data['img'].shape[1]
    max_size = max([int(np.sqrt(np.power(h, 2) + np.power(w, 2))),
                    int(w+h*np.cos(11/36)),
                    int(max(h,w)*1.2)
                    ]) + 5

    up = (max_size - h) // 2
    down = max_size - up - h
    left = (max_size - w) // 2
    right = max_size - left - w

    input_data['img'] = np.pad(input_data['img'], ((up, down), (left, right), (0, 0)), mode='constant')

    if input_data['is_text_cnts']:
        input_data['contour'] = list(
        map(lambda x: np.stack([x[:, :, 0] + up, x[:, :, 1] + left], axis=-1),  # x: np.array(n,1,2)
            input_data['contour']))
    else:
        input_data['contour'][0] = list(
            map(lambda x: np.stack([x[:, :, 0] + up, x[:, :, 1] + left], axis=-1),  # x: np.array(n,1,2)
                input_data['contour'][0]))
        input_data['contour'][1] = list(
            map(lambda x: np.stack([x[:, :, 0] + up, x[:, :, 1] + left], axis=-1),  # x: np.array(n,1,2)
                input_data['contour'][1]))

    input_data['center_point'] = list(
        map(lambda x: (x[0]+up, x[1]+left),
            input_data['center_point']))
    return input_data

def _pixel_augmentation(inputs):
    """
    1. pad a large black background
    2. doing augmentations that do not make affine transformation

    invert/add(overall/channel), add to hue and saturation, multiply,
    gaussian blur, average blur, median blur, bilateral blur,
    sharpen, emboss, edge detect,
    noise: guassian(overall/channel), dropout(pixel/channel, coarse,channel), salt&pepper
    norm: contrast
    gray scale

    return: padded + augmented image

    """
    input_data = copy.deepcopy(inputs)

    input_data['img'] = _get_seq().augment_image(input_data['img'])
    return input_data

def _affine_transformation(inputs, trans_rate=0.5):
    """
    affine types include:
        1. scaling
        2. rotate
        3. shear
        4. aspect ratios
    doing affine transformation
    :return:
    """
    input_data = copy.deepcopy(inputs)

    if random() > 1-trans_rate:
        transformer = _get_seq(affine=True)
        input_data['img'] = transformer.augment_image(input_data['img'])
        if input_data['is_text_cnts']:
            for p, cnt in enumerate(input_data['contour']):
                input_data['contour'][p] = transformer.augment_keypoints([
                    _key_points(image_shape=input_data['img'].shape, point_list=cnt)
                ])[0]
                input_data['contour'][p] = [np.array([[int(keypoints.y),
                                                       int(keypoints.x)]])[:, ::-1]
                                            for keypoints in input_data['contour'][p].keypoints]
                input_data['contour'][p] = np.stack(input_data['contour'][p], axis=0)
        else:
            for t in range(2):
                for p, cnt in enumerate(input_data['contour'][t]):
                    input_data['contour'][t][p] = transformer.augment_keypoints([
                        _key_points(image_shape=input_data['img'].shape, point_list=cnt)
                    ])[0]
                    input_data['contour'][t][p] = [np.array([[int(keypoints.y),
                                                           int(keypoints.x)]])[:, ::-1]
                                                for keypoints in input_data['contour'][t][p].keypoints]
                    input_data['contour'][t][p] = np.stack(input_data['contour'][t][p], axis=0)

        input_data['center_point'][0] = \
            transformer.augment_keypoints([
                _key_points(image_shape=input_data['img'].shape,
                                 point_list=np.array(
                                     [[list(input_data['center_point'])[0]]]
                                 ))
            ])[0].keypoints[0]
        input_data['center_point'][0] = (int(input_data['center_point'][0].y),
                                         int(input_data['center_point'][0].x))[::-1]
        input_data['center_point'][1] = transformer.augment_keypoints([
                _key_points(image_shape=input_data['img'].shape,
                                 point_list=np.array(
                                     [[list(input_data['center_point'])[1]]]
                                 ))

            ])[0].keypoints[0]
        input_data['center_point'][1] = (int(input_data['center_point'][1].y),
                                         int(input_data['center_point'][1].x))[::-1]

    input_data['img'] = np.transpose(input_data['img'], axes=[1, 0, 2])  #???

    return input_data

def _crop_flip_pad(input_data):
    """
    flip, pad, crop, as final states.
    :param input_data:
    :return:
    """
    center1 = input_data['center_point'][0]
    center2 = input_data['center_point'][1]
    p = random()
    x = center1[0]+int(p*(center2[0]-center1[0]))
    y = center1[1]+int(p*(center2[1]-center1[1]))
    return x, y

def augment(input_, augment_rate=100, trans_rate=0.5,test_mode=False,real_test=False):
    """

    :param input_data:
           Dict{'img_name':str,   original_name
            'img':np.uint8,
            'contour':List[the contour of each text instance],
            'type': 'char' or 'tl',
            'is_text_cnts':if this is synthetext or not,
            'char'}
            contour:
            if is_text_cnts is True: list(ndarray), ndarray: dtype np.float32, shape [n, 1, 2], order(col, row)
            if is_text_cnts is False: list(list(ndarray), list(ndarray)), for [char_cnts, text_cnts]
    :param augment_rate
    :param trans_rate
    :return:
    Dict{'img_name':str,   original_name
        'img':np.uint8,
        'contour':List[the contour of each text instance],
        'type': 'char' or 'tl',
        'flag':if this is synthetext or not,
        'left_top': tuple (x, y), x is row, y is col, please be careful about the order,
             'right_bottom': tuple (x, y), x is row, y is col}
    """
    input_data=copy.deepcopy(input_)
    if (not input_data['is_text_cnts']):
        if len(input_data['contour'][0][0].shape)<3:
            for i in range(2):
                input_data['contour'][i] = [np.stack([cnt],axis=1) for cnt in input_data['contour'][i]]
                input_data['contour'][i] = [cnt[:, :, ::-1] for cnt in input_data['contour'][i]]
    else:
        if len(input_data['contour'][0].shape) < 3:
            input_data['contour'] = [np.stack([cnt], axis=1) for cnt in input_data['contour']]
        input_data['contour'] = [cnt[:, :, ::-1] for cnt in input_data['contour']]

    if input_data['img'].shape[0] < input_data['img'].shape[1]:
        input_data['center_point'] = [(input_data['img'].shape[0] // 2, input_data['img'].shape[0] // 2),
                                  (input_data['img'].shape[0] // 2, input_data['img'].shape[1] - input_data['img'].shape[0] // 2)]
    else:
        input_data['center_point'] = [(input_data['img'].shape[1] // 2, input_data['img'].shape[1] // 2),
                                  (input_data['img'].shape[0] - input_data['img'].shape[1] // 2, input_data['img'].shape[1] // 2)]

    if real_test:
        input_data = _resize_to_32x(input_data)
        input_data['contour'] = [cnt[:, :, ::-1] for cnt in input_data['contour']]
        return {
            **input_data,
            'left_top': (0, 0),
            'right_bottom': (input_data['img'].shape[0], input_data['img'].shape[1])
        }

    if test_mode:
        input_data=_resize_to_512(input_data)
        input_data['contour'] = [cnt[:, :, ::-1] for cnt in input_data['contour']]
        return {
            **input_data,
            'left_top': (0, 0),
            'right_bottom': (input_data['img'].shape[0], input_data['img'].shape[1])
        }

    input_data = _resize(input_data)




    if (not input_data['is_text_cnts']) or (random() < (1.0 / augment_rate)):
        center_point = _crop_flip_pad(input_data)
        if (not input_data['is_text_cnts']):
            for i in range(2):
                input_data['contour'][i] = [cnt[:, :, ::-1] for cnt in input_data['contour'][i]]
        else:
            input_data['contour'] = [cnt[:, :, ::-1] for cnt in input_data['contour']]
        x_top=max(0,center_point[0] - 256)
        y_left=max(0,center_point[1] - 256)
        return {
            **input_data,
            'left_top': (x_top, y_left),
            'right_bottom': (x_top+512, y_left+512)
        }

    else:
        input_data = _pad(input_data)
        input_data['img'] = np.transpose(input_data['img'], axes=[1, 0, 2])  # ？？？
        transformed = _affine_transformation(_pixel_augmentation(input_data), trans_rate=trans_rate)
        center_point = _crop_flip_pad(transformed)
        transformed['contour'] = [cnt[:, :, ::-1] for cnt in transformed['contour']]
        x_top = max(0, center_point[0] - 256)
        y_left = max(0, center_point[1] - 256)
        return {**transformed,
                'left_top': (x_top, y_left),
                'right_bottom': (x_top+512, y_left+512)
        }

