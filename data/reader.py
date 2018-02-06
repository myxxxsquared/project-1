import scipy.io as sio
import numpy as np
import cv2
import os
import math
import tensorflow as tf

SYNTHTEXT_DIR = '/home/rjq/data/SynthText/SynthText/'
TOTALTEXT_DIR = '/home/rjq/data/Total-Text-Dataset/Download/'
MSRA_DIR ='/home/rjq/data/MSRA-TD500/MSRA-TD500/MSRA-TD500/'

TFRECORD_DIR='/home/rjq/data_cleaned/tfrecord/'


##generators:
def validate(im, cnts):
    cols, rows = [], []
    for cnt in cnts:
        for i in range(len(cnt)):
            cols.append(cnt[i][0][0])
            rows.append(cnt[i][0][1])
    col_max = math.ceil(max(cols))
    row_max = math.ceil(max(rows))
    im_row, im_col = im.shape[0]-1, im.shape[1]-1
    if im_row < row_max:
        temp = np.zeros([row_max-im_row, im.shape[1], im.shape[2]])
        im = np.concatenate((im, temp), 0)
    if im_col < col_max:
        temp = np.zeros([im.shape[0], col_max-im_col, im.shape[2]])
        im = np.concatenate((im, temp), 1)
    return im, cnts


def SynthText_loader(patch_num, n_th_patch):
    '''
    :param patch_num:
    :param n_th_patch:
    :param is_train:
    :return:
    '''
    gt = sio.loadmat(SYNTHTEXT_DIR+'gt.mat')
    pic_num = len(gt['imnames'][0])
    print(pic_num)
    patch_length = pic_num//patch_num+1
    start_point = n_th_patch*patch_length
    if (n_th_patch+1)*patch_length > pic_num:
        end_point = pic_num
    else:
        end_point = (n_th_patch+1)*patch_length
    print(start_point,end_point)
    for index in range(start_point, end_point):
        imname = gt['imnames'][0][index][0]
        origin = cv2.imread('/home/rjq/data/SynthText/SynthText/'+imname)
        origin = np.array(origin, np.uint8)
        assert origin.shape[2] == 3

        word_cnts = gt['wordBB'][0][index]
        char_cnts = gt['charBB'][0][index]
        if len(word_cnts.shape) == 2:
            word_cnts = np.expand_dims(word_cnts, 2)
        if len(char_cnts.shape) == 2:
            char_cnts = np.expand_dims(char_cnts, 2)
        word_cnts = np.transpose(word_cnts, (2,1,0))
        char_cnts = np.transpose(char_cnts, (2,1,0))

        char_cnts = [np.array(char_cnt, np.float32) for char_cnt in char_cnts]
        word_cnts = [np.array(word_cnt, np.float32) for word_cnt in word_cnts]
        cnts = [char_cnts, word_cnts]
        txt = gt['txt'][0][index].tolist()
        txt = [text.strip() for text in txt]

        chars = []
        for line in txt:
            for sub_line in line.split():
                temp = []
                for char in list(sub_line):
                    if char not in ('\n',):
                        temp.append(char)
                chars.append(temp)

        yield {'img_index': index,
               'img_name': imname,
               'img': origin,
               'contour': cnts,
               'chars': chars}


def Totaltext_loader(patch_num, n_th_patch, is_train):
    '''
    :param patch_num:
    :param n_th_patch:
    :param is_train: bool
    :return:
    '''
    def get_total_cnts(mat):
        cnts = []
        for i in range(len(mat['polygt'])):
            temp = []
            for x, y in zip(mat['polygt'][i][1][0], mat['polygt'][i][3][0]):
                temp.append([x,y])
            temp = np.expand_dims(np.array(temp), 1).astype(np.float32)
            cnts.append(temp)
        cnts_ = []
        for cnt in cnts:
            if len(cnt) >= 3:
                cnts_.append(np.array(cnt, np.float32))
        return cnts_

    if is_train:
        imnames = [name.split('.')[0] for name in os.listdir(TOTALTEXT_DIR + 'totaltext/Images/Train')]
        imnames = sorted(imnames)
        pic_num = len(imnames)
        patch_length = pic_num // patch_num+1
        start_point = n_th_patch * patch_length
        if (n_th_patch + 1) * patch_length > pic_num:
            end_point = pic_num
        else:
            end_point = (n_th_patch + 1) * patch_length

        for index in range(start_point, end_point):
            imname = imnames[index]
            origin = cv2.imread(TOTALTEXT_DIR+'totaltext/Images/Train/'+imname+'.jpg')
            if origin is None:
                origin = cv2.imread(TOTALTEXT_DIR+'totaltext/Images/Train/'+imname+'.JPG')
            if origin is None:
                print(imname+ ' is missed')
                continue
            mat = sio.loadmat(TOTALTEXT_DIR + 'groundtruth_text/Groundtruth/Polygon/Train/poly_gt_' + imname + '.mat')
            cnts = get_total_cnts(mat)
            origin, cnts = validate(origin, cnts)
            origin = np.array(origin, np.uint8)
            cnts = [np.array(cnt, np.float32) for cnt in cnts]
            yield {'img_index': index,
                   'img_name': imname,
                   'img': origin,
                   'contour': cnts}

    else:
        imnames = [name.split('.')[0] for name in os.listdir(TOTALTEXT_DIR + 'totaltext/Images/Test')]
        imnames = sorted(imnames)
        pic_num = len(imnames)
        patch_length = pic_num//patch_num+1
        start_point = n_th_patch*patch_length
        if (n_th_patch+1)*patch_length > pic_num:
            end_point = pic_num
        else:
            end_point = (n_th_patch+1)*patch_length

        for index in range(start_point, end_point):
            imname = imnames[index]
            origin = cv2.imread(TOTALTEXT_DIR+'totaltext/Images/Test/'+imname+'.jpg')
            if origin is None:
                origin = cv2.imread(TOTALTEXT_DIR+'totaltext/Images/Test/'+imname+'.JPG')
            if origin is None:
                print(imname + 'is missed')
                continue
            mat = sio.loadmat(TOTALTEXT_DIR + 'groundtruth_text/Groundtruth/Polygon/Test/poly_gt_' + imname + '.mat')
            cnts = get_total_cnts(mat)
            origin, cnts = validate(origin, cnts)
            origin = np.array(origin, np.uint8)
            cnts = [np.array(cnt, np.float32) for cnt in cnts]
            yield {'img_index': index,
                   'img_name': imname,
                   'img': origin,
                   'contour': cnts}


def MSRA_TD_500_loader(patch_num, n_th_patch, is_train):
    '''
    :param patch_num:
    :param n_th_patch:
    :param is_train:
    :return:
    '''
    def get_cnts_msra(textes):
        cnts = []
        def reverse_point(point):
            return (point[1], point[0])
        for text in textes:
            points = []
            text = [float(num) for num in text]
            x, y, w, h, theta = text[2], text[3], text[4], text[5], text[6]
            point1 = (x, y)
            point2 = (x+w, y)
            point3 = (x+w, y+h)
            point4 = (x, y+h)
            rotateMatrix = cv2.getRotationMatrix2D((x+w/2,y+h/2), -theta*180/np.pi,1)
            point1 = np.matmul(rotateMatrix, point1+(1,))
            point2 = np.matmul(rotateMatrix, point2+(1,))
            point3 = np.matmul(rotateMatrix, point3+(1,))
            point4 = np.matmul(rotateMatrix, point4+(1,))
            points.append([point1])
            points.append([point2])
            points.append([point3])
            points.append([point4])
            cnts.append(np.array(points).astype(np.int32))
        return cnts

    if is_train:
        imnames = list(set([name.split('.')[0] for name in os.listdir(MSRA_DIR + 'train/')]))
        imnames = sorted(imnames)
        pic_num = len(imnames)
        patch_length = pic_num//patch_num+1
        start_point = n_th_patch*patch_length
        if (n_th_patch+1)*patch_length > pic_num:
            end_point = pic_num
        else:
            end_point = (n_th_patch+1)*patch_length

        for index in range(start_point, end_point):
            print(index)
            imname = imnames[index]
            origin = cv2.imread(MSRA_DIR+'train/'+imname+'.JPG')
            if origin is None:
                print(imname + ' is missed')
                continue
            textes = [text.split() for text in open(MSRA_DIR+'train/'+imname+'.gt', 'r').readlines()]
            if len(textes) == 0:
                print('cnt for '+imname+'is missed')
                continue
            cnts = get_cnts_msra(textes)
            origin, cnts = validate(origin, cnts)
            yield {'img_index': index,
                   'img': origin,
                   'contour': cnts}

    else:
        imnames = list(set([name.split('.')[0] for name in os.listdir(MSRA_DIR + 'test/')]))
        imnames = sorted(imnames)
        pic_num = len(imnames)
        patch_length = pic_num // patch_num + 1
        start_point = n_th_patch * patch_length
        if (n_th_patch + 1) * patch_length > pic_num:
            end_point = pic_num
        else:
            end_point = (n_th_patch + 1) * patch_length

        for index in range(start_point, end_point):
            imname = imnames[index]
            origin = cv2.imread(MSRA_DIR + 'test/' + imname + '.JPG')
            if origin is None:
                print(imname + ' is missed')
                continue
            textes = [text.split() for text in open(MSRA_DIR + 'test/' + imname + '.gt', 'r').readlines()]
            if len(textes) == 0:
                print('cnt for ' + imname + 'is missed')
                continue
            cnts = get_cnts_msra(textes)
            origin, cnts = validate(origin, cnts)
            yield {'img_index': index,
                   'img': origin,
                   'contour': cnts}


def ICDAR2017_loader(start_point,end_point):
    """
    :param start_point:
    :param end_point:
    :return:
    {'img_name':str,
        'img':np.uint8,
        'contour':List[the contour of each text instance]}
    """
    pass


def ICDAR2015_loader(start_point,end_point):
    """
    :param start_point:
    :param end_point:
    :return:
    {'img_name':str,
        'img':np.uint8,
        'contour':List[the contour of each text instance]}
    """
    pass


def ICDAR2013_loader(start_point,end_point):
    """
    :param start_point:
    :param end_point:
    :return:
    {'img_name':str,
        'img':np.uint8,
        'contour':List[the contour of each text instance]}
    """
    pass


def TD500_loader(start_point,end_point):
    """
    :param start_point:
    :param end_point:
    :return:
    {'img_name':str,
        'img':np.uint8,
        'contour':List[the contour of each text instance]}
    """
    pass


    ############ codes below are for tfrecord ##############
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _pad_cnt(cnt, cnt_point_max):
    new = []
    for cnt_ in cnt:
        if len(cnt_) < cnt_point_max:
            new.append(np.concatenate((cnt_, np.zeros([cnt_point_max-len(cnt_), 1, 2])), 0))
        else:
            new.append(cnt_)
    return new


def othertext(save_dir, patch_num, n_th_patch, is_train, dataset):
    print('start')
    save_dir = save_dir.strip('/')
    save_dir = save_dir + '/'
    if not os.path.exists(TFRECORD_DIR+save_dir):
        os.mkdir(TFRECORD_DIR+save_dir)
    if is_train:
        tfrecords_filename = TFRECORD_DIR+save_dir+str(n_th_patch)+'_'+dataset+'_train.tfrecords'
    else:
        tfrecords_filename = TFRECORD_DIR+save_dir+str(n_th_patch)+'_'+dataset+'_test.tfrecords'

    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    print('get writer')
    count = 0
    generators = {'totaltext': Totaltext_loader}
                  # 'msra': MSRA_TD_500_loader}
    generator = generators[dataset]
    print(generator)

    for res in generator(patch_num, n_th_patch, is_train):
        count += 1
        print('processing ' +str(count))
        img_index = res['img_index']
        img = res['img']
        img = np.array(img, np.uint8)
        img_row = img.shape[0]
        img_col = img.shape[1]
        contour = res['contour']
        cnt_point_num = np.array([len(contour[i]) for i in range(len(contour))], np.int64)
        cnt_num = len(contour)
        cnt_point_max = int(max(cnt_point_num))
        contour = _pad_cnt(contour, cnt_point_max)
        contour = np.array(contour, np.float32)
        example = tf.train.Example(features=tf.train.Features(feature={
            'img_index': _int64_feature(img_index),
            'img': _bytes_feature(img.tostring()),
            'contour': _bytes_feature(contour.tostring()),
            'im_row': _int64_feature(img_row),
            'im_col': _int64_feature(img_col),
            'cnt_num': _int64_feature(cnt_num),
            'cnt_point_num': _bytes_feature(cnt_point_num.tostring()),
            'cnt_point_max': _int64_feature(cnt_point_max)
        }))
        writer.write(example.SerializeToString())
    writer.close()
    #
    # def synthtext(save_dir, patch_num, n_th_patch):
    #     save_dir = save_dir.strip('/')
    #     save_dir = save_dir + '/'
    #     if not os.path.exists(TFRECORD_DIR+save_dir):
    #         os.mkdir(TFRECORD_DIR+save_dir)
    #
    #     tfrecords_filename = TFRECORD_DIR+save_dir+str(n_th_patch)+'_synthtext.tfrecords'
    #     writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    #     count = 0
    #     for res in SynthText_loader(patch_num, n_th_patch, True):
    #         count += 1
    #         print('processing ' +str(count))
    #         img_index = res['img_index']
    #         img = res['img']
    #         img = np.array(img, np.uint8)
    #         img_row = img.shape[0]
    #         img_col = img.shape[1]
    #         contour = res['contour']
    #         char_contour, word_contour = contour
    #
    #         char_cnt_point_num = np.array([len(char_contour[i]) for i in range(len(char_contour))], np.int64)
    #         char_cnt_num = len(char_contour)
    #         char_cnt_point_max = int(max(char_cnt_point_num))
    #         char_contour = _pad_cnt(char_contour, char_cnt_point_max)
    #         char_contour = np.array(char_contour, np.float32)
    #
    #         word_cnt_point_num = np.array([len(word_contour[i]) for i in range(len(word_contour))], np.int64)
    #         word_cnt_num = len(word_contour)
    #         word_cnt_point_max = int(max(word_cnt_point_num))
    #         word_contour = _pad_cnt(word_contour, word_cnt_point_max)
    #         word_contour = np.array(word_contour, np.float32)
    #
    #
    #         example = tf.train.Example(features=tf.train.Features(feature={
    #             'img_index': _int64_feature(img_index),
    #             'img': _bytes_feature(img.tostring()),
    #             'char_contour': _bytes_feature(char_contour.tostring()),
    #             'word_contour': _bytes_feature(word_contour.tostring()),
    #             'im_row': _int64_feature(img_row),
    #             'im_col': _int64_feature(img_col),
    #             'char_cnt_num': _int64_feature(char_cnt_num),
    #             'char_cnt_point_num': _bytes_feature(char_cnt_point_num.tostring()),
    #             'char_cnt_point_max': _int64_feature(char_cnt_point_max),
    #             'word_cnt_num': _int64_feature(word_cnt_num),
    #             'word_cnt_point_num': _bytes_feature(word_cnt_point_num.tostring()),
    #             'word_cnt_point_max': _int64_feature(word_cnt_point_max)
    #
    #         }))
    #         writer.write(example.SerializeToString())
    #     writer.close()
    #
    # def othertext_decoder(tfrecords_filename):
    #     record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
    #     for string_record in record_iterator:
    #         example = tf.train.Example()
    #         example.ParseFromString(string_record)
    #
    #         img_index = int(example.features.feature['img_index']
    #                      .int64_list
    #                      .value[0])
    #         img_string = (example.features.feature['img']
    #                         .bytes_list
    #                         .value[0])
    #         contour_string = (example.features.feature['contour']
    #                         .bytes_list
    #                         .value[0])
    #         img_row = int(example.features.feature['im_row']
    #                      .int64_list
    #                      .value[0])
    #         img_col = int(example.features.feature['im_col']
    #                      .int64_list
    #                      .value[0])
    #         cnt_num = int(example.features.feature['cnt_num']
    #                      .int64_list
    #                      .value[0])
    #         cnt_point_num_string = (example.features.feature['cnt_point_num']
    #                         .bytes_list
    #                         .value[0])
    #         cnt_point_max = int(example.features.feature['cnt_point_max']
    #                      .int64_list
    #                      .value[0])
    #
    #         img_1d = np.fromstring(img_string, dtype=np.uint8)
    #         reconstructed_img = img_1d.reshape((img_row, img_col, -1))
    #         img = reconstructed_img
    #         cnt_point_num = np.fromstring(cnt_point_num_string, dtype=np.int64)
    #
    #         contour_1d = np.fromstring(contour_string, dtype=np.float32)
    #         reconstructed_contour = contour_1d.reshape((cnt_num, cnt_point_max, 1, 2))
    #         contour = []
    #         for i in range(cnt_num):
    #             contour.append(reconstructed_contour[i, :cnt_point_num[i], :, :])
    #         yield {'img_index': img_index,
    #                'img': img,
    #                'contour': contour}
    #
    # def synthtext_decoder(tfrecords_filename):
    #     record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
    #     for string_record in record_iterator:
    #         example = tf.train.Example()
    #         example.ParseFromString(string_record)
    #
    #         img_index = int(example.features.feature['img_index']
    #                         .int64_list
    #                         .value[0])
    #         img_string = (example.features.feature['img']
    #             .bytes_list
    #             .value[0])
    #         char_contour_string = (example.features.feature['char_contour']
    #             .bytes_list
    #             .value[0])
    #         word_contour_string = (example.features.feature['word_contour']
    #             .bytes_list
    #             .value[0])
    #         img_row = int(example.features.feature['im_row']
    #                       .int64_list
    #                       .value[0])
    #         img_col = int(example.features.feature['im_col']
    #                       .int64_list
    #                       .value[0])
    #         char_cnt_num = int(example.features.feature['char_cnt_num']
    #                       .int64_list
    #                       .value[0])
    #         char_cnt_point_num_string = (example.features.feature['char_cnt_point_num']
    #             .bytes_list
    #             .value[0])
    #         char_cnt_point_max = int(example.features.feature['char_cnt_point_max']
    #                             .int64_list
    #                             .value[0])
    #         word_cnt_num = int(example.features.feature['word_cnt_num']
    #                       .int64_list
    #                       .value[0])
    #         word_cnt_point_num_string = (example.features.feature['word_cnt_point_num']
    #             .bytes_list
    #             .value[0])
    #         word_cnt_point_max = int(example.features.feature['word_cnt_point_max']
    #                             .int64_list
    #                             .value[0])
    #
    #         img_1d = np.fromstring(img_string, dtype=np.uint8)
    #         reconstructed_img = img_1d.reshape((img_row, img_col, -1))
    #         img = reconstructed_img
    #
    #         char_cnt_point_num = np.fromstring(char_cnt_point_num_string, dtype=np.int64)
    #         char_contour_1d = np.fromstring(char_contour_string, dtype=np.float32)
    #         char_reconstructed_contour = char_contour_1d.reshape((char_cnt_num, char_cnt_point_max, 1, 2))
    #         char_contour = []
    #         for i in range(char_cnt_num):
    #             char_contour.append(char_reconstructed_contour[i, :char_cnt_point_num[i], :, :])
    #
    #         word_cnt_point_num = np.fromstring(word_cnt_point_num_string, dtype=np.int64)
    #         word_contour_1d = np.fromstring(word_contour_string, dtype=np.float32)
    #         word_reconstructed_contour = word_contour_1d.reshape((word_cnt_num, word_cnt_point_max, 1, 2))
    #         word_contour = []
    #         for i in range(word_cnt_num):
    #             word_contour.append(word_reconstructed_contour[i, :word_cnt_point_num[i], :, :])
    #
    #         yield {'img_index': img_index,
    #                'img': img,
    #                'contour': [char_contour, word_contour]}


def synthtext(save_dir, patch_num, n_th_patch):
    save_dir = save_dir.strip('/')
    save_dir = save_dir + '/'
    if not os.path.exists(TFRECORD_DIR+save_dir):
        os.mkdir(TFRECORD_DIR+save_dir)

    tfrecords_filename = TFRECORD_DIR+save_dir+str(n_th_patch)+'_synthtext.tfrecords'
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    count = 0
    for res in SynthText_loader(patch_num, n_th_patch):
        count += 1
        print('processing ' +str(count))
        img_index = res['img_index']
        img = res['img']
        img = np.array(img, np.uint8)
        img_row = img.shape[0]
        img_col = img.shape[1]
        contour = res['contour']
        char_contour, word_contour = contour

        char_cnt_point_num = np.array([len(char_contour[i]) for i in range(len(char_contour))], np.int64)
        char_cnt_num = len(char_contour)
        char_cnt_point_max = int(max(char_cnt_point_num))
        char_contour = _pad_cnt(char_contour, char_cnt_point_max)
        char_contour = np.array(char_contour, np.float32)

        word_cnt_point_num = np.array([len(word_contour[i]) for i in range(len(word_contour))], np.int64)
        word_cnt_num = len(word_contour)
        word_cnt_point_max = int(max(word_cnt_point_num))
        word_contour = _pad_cnt(word_contour, word_cnt_point_max)
        word_contour = np.array(word_contour, np.float32)


        example = tf.train.Example(features=tf.train.Features(feature={
            'img_index': _int64_feature(img_index),
            'img': _bytes_feature(img.tostring()),
            'char_contour': _bytes_feature(char_contour.tostring()),
            'word_contour': _bytes_feature(word_contour.tostring()),
            'im_row': _int64_feature(img_row),
            'im_col': _int64_feature(img_col),
            'char_cnt_num': _int64_feature(char_cnt_num),
            'char_cnt_point_num': _bytes_feature(char_cnt_point_num.tostring()),
            'char_cnt_point_max': _int64_feature(char_cnt_point_max),
            'word_cnt_num': _int64_feature(word_cnt_num),
            'word_cnt_point_num': _bytes_feature(word_cnt_point_num.tostring()),
            'word_cnt_point_max': _int64_feature(word_cnt_point_max)

        }))
        writer.write(example.SerializeToString())
    writer.close()


def othertext_decoder(tfrecords_filename):
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        img_index = int(example.features.feature['img_index']
                     .int64_list
                     .value[0])
        img_string = (example.features.feature['img']
                        .bytes_list
                        .value[0])
        contour_string = (example.features.feature['contour']
                        .bytes_list
                        .value[0])
        img_row = int(example.features.feature['im_row']
                     .int64_list
                     .value[0])
        img_col = int(example.features.feature['im_col']
                     .int64_list
                     .value[0])
        cnt_num = int(example.features.feature['cnt_num']
                     .int64_list
                     .value[0])
        cnt_point_num_string = (example.features.feature['cnt_point_num']
                        .bytes_list
                        .value[0])
        cnt_point_max = int(example.features.feature['cnt_point_max']
                     .int64_list
                     .value[0])

        img_1d = np.fromstring(img_string, dtype=np.uint8)
        reconstructed_img = img_1d.reshape((img_row, img_col, -1))
        img = reconstructed_img
        cnt_point_num = np.fromstring(cnt_point_num_string, dtype=np.int64)

        contour_1d = np.fromstring(contour_string, dtype=np.float32)
        reconstructed_contour = contour_1d.reshape((cnt_num, cnt_point_max, 1, 2))
        contour = []
        for i in range(cnt_num):
            contour.append(reconstructed_contour[i, :cnt_point_num[i], :, :])
        yield {'img_index': img_index,
               'img': img,
               'contour': contour}


def synthtext_decoder(tfrecords_filename):
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        img_index = int(example.features.feature['img_index']
                        .int64_list
                        .value[0])
        img_string = (example.features.feature['img']
            .bytes_list
            .value[0])
        char_contour_string = (example.features.feature['char_contour']
            .bytes_list
            .value[0])
        word_contour_string = (example.features.feature['word_contour']
            .bytes_list
            .value[0])
        img_row = int(example.features.feature['im_row']
                      .int64_list
                      .value[0])
        img_col = int(example.features.feature['im_col']
                      .int64_list
                      .value[0])
        char_cnt_num = int(example.features.feature['char_cnt_num']
                      .int64_list
                      .value[0])
        char_cnt_point_num_string = (example.features.feature['char_cnt_point_num']
            .bytes_list
            .value[0])
        char_cnt_point_max = int(example.features.feature['char_cnt_point_max']
                            .int64_list
                            .value[0])
        word_cnt_num = int(example.features.feature['word_cnt_num']
                      .int64_list
                      .value[0])
        word_cnt_point_num_string = (example.features.feature['word_cnt_point_num']
            .bytes_list
            .value[0])
        word_cnt_point_max = int(example.features.feature['word_cnt_point_max']
                            .int64_list
                            .value[0])

        img_1d = np.fromstring(img_string, dtype=np.uint8)
        reconstructed_img = img_1d.reshape((img_row, img_col, -1))
        img = reconstructed_img

        char_cnt_point_num = np.fromstring(char_cnt_point_num_string, dtype=np.int64)
        char_contour_1d = np.fromstring(char_contour_string, dtype=np.float32)
        char_reconstructed_contour = char_contour_1d.reshape((char_cnt_num, char_cnt_point_max, 1, 2))
        char_contour = []
        for i in range(char_cnt_num):
            char_contour.append(char_reconstructed_contour[i, :char_cnt_point_num[i], :, :])

        word_cnt_point_num = np.fromstring(word_cnt_point_num_string, dtype=np.int64)
        word_contour_1d = np.fromstring(word_contour_string, dtype=np.float32)
        word_reconstructed_contour = word_contour_1d.reshape((word_cnt_num, word_cnt_point_max, 1, 2))
        word_contour = []
        for i in range(word_cnt_num):
            word_contour.append(word_reconstructed_contour[i, :word_cnt_point_num[i], :, :])

        yield {'img_index': img_index,
               'img': img,
               'contour': [char_contour, word_contour]}


#########codes below are testing tfrecord###########
def othertext_to_tf(save_dir, patch_num, n_th_patch, is_train, dataset):

    save_dir = save_dir.strip('/')
    save_dir = save_dir + '/'
    if not os.path.exists(TFRECORD_DIR+save_dir):
        os.mkdir(TFRECORD_DIR+save_dir)
    if is_train:
        tfrecords_filename = TFRECORD_DIR+save_dir+str(n_th_patch)+'_'+dataset+'_train.tfrecords'
    else:
        tfrecords_filename = TFRECORD_DIR+save_dir+str(n_th_patch)+'_'+dataset+'_test.tfrecords'

    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    print('get writer')
    count = 0
    generators = {'totaltext': Totaltext_loader}
                  # 'msra': MSRA_TD_500_loader}
    generator = generators[dataset]
    print(generator)

    for res in generator(patch_num, n_th_patch, is_train):
        count += 1
        print('processing ' +str(count))
        img_index = res['img_index']
        img = res['img']
        img = np.array(img, np.uint8)
        img_row = img.shape[0]
        img_col = img.shape[1]
        contour = res['contour']
        cnt_point_num = np.array([len(contour[i]) for i in range(len(contour))], np.int64)
        cnt_num = len(contour)
        cnt_point_max = int(max(cnt_point_num))
        contour = _pad_cnt(contour, cnt_point_max)
        contour = np.array(contour, np.float32)
        example = tf.train.Example(features=tf.train.Features(feature={
            'img_index': _int64_feature(img_index),
            'img': _bytes_feature(img.tostring()),
            'contour': _bytes_feature(contour.tostring()),
            'im_row': _int64_feature(img_row),
            'im_col': _int64_feature(img_col),
            'cnt_num': _int64_feature(cnt_num),
            'cnt_point_num': _bytes_feature(cnt_point_num.tostring()),
            'cnt_point_max': _int64_feature(cnt_point_max)
        }))
        writer.write(example.SerializeToString())
    writer.close()



if __name__ == '__main__':
    import pickle
    from multiprocessing import Pool
    from multiprocessing import Process

    PKL_DIR = '/home/rjq/data_cleaned/pkl/'
    generators = {'totaltext': Totaltext_loader}


    def othertext_to_pickle(save_dir, patch_num, n_th_patch, is_train, dataset):
        save_dir = save_dir.strip('/')
        save_dir = save_dir + '/'
        if not os.path.exists(PKL_DIR+save_dir):
           os.mkdir(PKL_DIR+save_dir)
        save_path = PKL_DIR+save_dir
        count = 0
        generator = generators[dataset]

        for res in generator(patch_num, n_th_patch, is_train):
            count += 1
            print('processing ' +str(count))
            img_index = res['img_index']
            img_name = res['img_name']
            img = res['img']
            contour = res['contour']
            img = np.array(img, np.uint8)
            contour = [np.array(cnt, np.float32) for cnt in contour]

            data_instance={
                'img_name':img_name,
                'img':img,
                'contour':contour,
                'is_text_cnts': True
            }

            pickle.dump(data_instance,open(os.path.join(save_path,'{}.bin'.format((img_index))),'wb'))


    def synthtext_to_pickle(save_dir, patch_num, n_th_patch):
        save_dir = save_dir.strip('/')
        save_dir = save_dir + '/'
        if not os.path.exists(PKL_DIR+save_dir):
            os.mkdir(PKL_DIR + save_dir)
        save_path = PKL_DIR + save_dir

        count = patch_num*n_th_patch
        for res in SynthText_loader(patch_num, n_th_patch):
            count += 1
            print('processing ' +str(count))
            img_index = res['img_index']
            img_name = res['img_name']
            img = res['img']
            contour = res['contour']
            chars = res['chars']
            char_contour, word_contour = contour
            img = np.array(img, np.uint8)
            char_contour = np.array(char_contour, np.float32)
            word_contour = np.array(word_contour, np.float32)
            contour = [char_contour, word_contour]

            data_instance = {
                'img_name': img_name,
                'img': img,
                'contour': contour,
                'is_text_cnts': False,
                'chars': chars
            }

            pickle.dump(data_instance, open(os.path.join(save_path, '{}.bin'.format((img_index))), 'wb'))


    patch_num = 20
    p=Pool(patch_num)
    # p.apply_async(othertext_to_pickle, args=('totaltext_train/', 1, 0, True, 'totaltext'))
    # p.apply_async(othertext_to_pickle, args=('totaltext_test/', 1, 0, False, 'totaltext'))
    for i in range(patch_num):
        p.apply_async(synthtext_to_pickle,args=('synthtext_chars/', patch_num, i))
    p.close()
    p.join()


