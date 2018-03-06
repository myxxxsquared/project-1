from data.utils import get_maps
import numpy as np
import cv2
import random
import time

# class data_churn(object):
#     def __init__(self, thickness=0.2, neighbor=7.0, crop_skel=1.0, *args, **kw):
#         """
#         initialize an instance
#         :param kw: 'data_set': str, 'SynthText', 'totaltext', etc.
#                      'start_point','end_point':int, indicating the starting point for the crunching process
#                thickness: the thickness of the text center line
#                neighbor: the range used for fit the theta
#                crop_skel: the length for cropping the text center line (skeleton)
#         """
#         self.thickness = thickness
#         self.neighbor = neighbor
#         self.crop_skel = crop_skel
#         pass
#
#     def _data_labeling(self, img_name, img, cnts, is_text_cnts, left_top, right_bottom, chars, care):
#         '''
#         :param img_name: pass to return directly, (to be determined, int or str)
#         :param img: ndarray, np.uint8,
#         :param cnts:
#                 if is_text_cnts is True: list(ndarray), ndarray: dtype np.float32, shape [n, 1, 2], order(col, row)
#                 if is_text_cnts is False: list(list(ndarray), list(ndarray)), for [char_cnts, text_cnts]
#         :param is_text_cnts: bool
#         :param left_top: for cropping
#         :param right_bottom: for cropping
#         :param chars:
#                 if is_text_cnts is True: None
#                 if is_text_cnts is False: a nested list storing the chars info for synthtext
#         :return:
#                 img_name: passed down
#                 img: np.ndarray np.uint8
#                 maps: label map with a size of (512, 512, 5) aranged as [TCL, radius, cos_theta, sin_theta, TR], all of them are 2-d array,
#                 TR: np.bool; TCL: np.bool; radius: np.float32; cos_theta/sin_theta: np.float32
#         '''
#         try:
#             #t = time.time()
#             skels_points, radius_dict, score_dict, cos_theta_dict, sin_theta_dict, mask_fills = \
#                 get_maps(img, cnts, is_text_cnts, self.thickness, self.crop_skel, self.neighbor, char)
#             #print(img_name,'label takes:', time.time() - t)
#             #t = time.time()
#             TR = mask_fills[0]
#             for i in range(1, len(mask_fills)):
#                 TR = np.bitwise_or(TR, mask_fills[i])
#             TCL = np.zeros(img.shape[:2], np.bool)
#             for point, _ in score_dict.items():
#                 TCL[point[0], point[1]] = True
#             radius = np.zeros(img.shape[:2], np.float32)
#             for point, r in radius_dict.items():
#                 radius[point[0], point[1]] = r
#             cos_theta = np.zeros(img.shape[:2], np.float32)
#             for point, c_t in cos_theta_dict.items():
#                 cos_theta[point[0], point[1]] = c_t
#             sin_theta = np.zeros(img.shape[:2], np.float32)
#             for point, s_t in sin_theta_dict.items():
#                 sin_theta[point[0], point[1]] = s_t
#             TR = TR[left_top[0]:right_bottom[0], left_top[1]:right_bottom[1]]
#             TCL = TCL[left_top[0]:right_bottom[0], left_top[1]:right_bottom[1]]
#             radius = radius[left_top[0]:right_bottom[0], left_top[1]:right_bottom[1]]
#             cos_theta = cos_theta[left_top[0]:right_bottom[0], left_top[1]:right_bottom[1]]
#             sin_theta = sin_theta[left_top[0]:right_bottom[0], left_top[1]:right_bottom[1]]
#             img = img[left_top[0]:right_bottom[0], left_top[1]:right_bottom[1], :]
#             maps = [TCL, radius, cos_theta, sin_theta, TR]
#             #print(img_name,'mapping takes:', time.time() - t)
#             return img_name, img, np.stack(maps, -1), cnts
#         except:
#             print('Warning: error encountered in %s' % img_name)
#             return None, None, None, None
#
#     def _pixellink_labeling(self, img_name, img, cnts, is_text_cnts, left_top, right_bottom, chars, care):
#
#         # cv2.imwrite('img.jpg', img)
#         # cv2.imwrite('cnts.jpg', cv2.drawContours(img, cnts, -1,255,1))
#
#         map_shape = (img.shape[0]//2, img.shape[1]//2)
#
#         cnts = [cnt//2 for cnt in cnts]
#
#         # mask
#         mask = cv2.fillPoly(np.zeros(map_shape), cnts, 255)
#         mask = np.sign(mask).astype(np.float32)
#
#         # links & weight
#         links = [np.zeros(map_shape, np.float32) for _ in range(8)]
#         weight = np.zeros(map_shape, np.float32)
#
#         def _move(cnt, dir):
#             '''
#             0 1 2
#             3 x 4
#             5 6 7
#             :param cnt: np shape (n, 1, 2) ord: (col, row)
#             :param direct:
#             :return:
#             '''
#             directs = [(-1,-1), (0,-1), (1,-1), (-1,0), (1,0), (-1,1),(0,1),(1,1)]
#             return cnt+directs[dir]
#
#         for cnt_index in range(len(cnts)):
#             base = cv2.fillPoly(np.zeros(map_shape), [cnts[cnt_index]], 255).astype(np.bool)
#             temp = base > 0
#             weight[temp] = 1/np.sum(temp)/len(cnts)
#             # print(1/np.sum(temp)/len(cnts))
#             for i in range(8):
#                 mask_ = cv2.fillPoly(np.zeros(map_shape), [_move(cnts[cnt_index], 7-i)], 255).astype(np.bool)
#                 links[i][base&mask_] = 1.0
#
#         # for i in range(8):
#         #     cv2.imwrite('img'+str(i)+'.jpg', links[i]*255)
#
#         maps = np.stack([mask]+ links+[weight], -1)
#         img = img[left_top[0]:right_bottom[0], left_top[1]:right_bottom[1], :]
#         left_top = [left_top[0]//2, left_top[1]//2]
#         right_bottom = [right_bottom[0]//2,right_bottom[1]//2]
#         maps = maps[left_top[0]:right_bottom[0], left_top[1]:right_bottom[1], :]
#         return img_name, img, cnts, maps


def _pixellink_labeling(img_name, img, cnts, left_top, right_bottom):
    map_shape = (img.shape[0]//2, img.shape[1]//2)
    cnts = [cnt//2 for cnt in cnts]

    # mask
    mask = cv2.fillPoly(np.zeros(map_shape), cnts, 255)
    mask = np.sign(mask).astype(np.float32)

    # links & weight
    links = [np.zeros(map_shape, np.float32) for _ in range(8)]
    weight = np.zeros(map_shape, np.float32)

    def _move(cnt, dir):
        '''
        0 1 2
        3 x 4
        5 6 7
        :param cnt: np shape (n, 1, 2) ord: (col, row)
        :param direct:
        :return:
        '''
        directs = [(-1,-1), (0,-1), (1,-1), (-1,0), (1,0), (-1,1),(0,1),(1,1)]
        return cnt+directs[dir]

    for cnt_index in range(len(cnts)):
        base = cv2.fillPoly(np.zeros(map_shape), [cnts[cnt_index]], 255).astype(np.bool)
        temp = base > 0
        weight[temp] = 1/np.sum(temp)/len(cnts)
        # print(1/np.sum(temp)/len(cnts))
        for i in range(8):
            mask_ = cv2.fillPoly(np.zeros(map_shape), [_move(cnts[cnt_index], 7-i)], 255).astype(np.bool)
            links[i][base&mask_] = 1.0


    maps = np.stack([mask]+ links+[weight], -1)
    img = img[left_top[0]:right_bottom[0], left_top[1]:right_bottom[1], :]
    left_top = [left_top[0]//2, left_top[1]//2]
    right_bottom = [right_bottom[0]//2,right_bottom[1]//2]
    maps = maps[left_top[0]:right_bottom[0], left_top[1]:right_bottom[1], :]
    return img_name, img, cnts, maps



def _pixellink_transform(ins):
    img = ins['img']
    contour = ins['contour']
    img_name = ins['img_name']

    # step1: rotate
    random.seed(time.time())
    def _rotate(img, contour):
        row, col = img.shape[:2]
        img = np.rot90(img)
        new = []
        for cnt in contour:
            points = []
            for point in cnt:
                # y is col, x is row
                y,x = point[0]
                new_y, new_x = x, col-y
                points.append([[new_y,new_x]])
            points = np.array(points, np.float32)
            new.append(points)
        return img, new

    if random.random() <= 0.2:
        # counter clock, 0:0, 1: 90; 2: 180, 3: 270,
        rotate_time = random.randint(0,3)
        for i in range(rotate_time):
            img, contour = _rotate(img, contour)

    # step2: resize and aspect
    new_row, new_col=0,0
    row, col = img.shape[:2]
    while new_row-512-1 <0 or new_col-512-1 <0:
        size_ratio = random.random()* 2
        aspect_ratio = random.random()*1.5+0.5
        new_row, new_col = row*size_ratio, col*size_ratio
        new_row, new_col = new_row, new_col*aspect_ratio
        new_row, new_col = int(new_row), int(new_col)

    print('sampled resize and aspect')
    img = cv2.resize(img, (new_row, new_col))
    new = []
    for cnt in contour:
        points = []
        for point in cnt:
            # y is col, x is row
            y, x = point[0]
            new_y, new_x = y*col/new_col, x*row/new_row
            points.append([[new_y, new_x]])
        points = np.array(points, np.float32)
        new.append(points)
    contour = new

    # step3: get crop points
    left_top = random.randint(0,new_row-512-1), random.randint(0,new_col-512-1)
    right_bottom = left_top[0]+512, left_top[1]+512

    return img_name, img, contour, left_top, right_bottom


def pixellink_prepro(ins):
    img_name, img, cnts, maps = _pixellink_labeling(_pixellink_transform(ins))
    return img_name, img, cnts, maps


if __name__ == '__main__':
    import os
    import pickle
    TOTAL_TRAIN_DIR = '/home/rjq/data_cleaned/pkl/totaltext_train_care/'
    TOTAL_TEST_DIR = '/home/rjq/data_cleaned/pkl/totaltext_test_care/'

    file_names_totaltext_train = [TOTAL_TRAIN_DIR+name for name in os.listdir(TOTAL_TRAIN_DIR)]

    for file_name in file_names_totaltext_train:
        ins = pickle.load(open(file_name, 'rb'))
        img = ins['img'].copy()
        cnts = [cnt.astype(np.int32) for cnt in ins['contour']]
        img = cv2.drawContours(img, cnts,-1,(255,0,0), 1)
        cv2.imwrite('origin.jpg', img)
        img_name, img, cnts, maps = pixellink_prepro(ins)
        cnts = [cnt.astype(np.int32) for cnt in cnts]
        img = cv2.drawContours(img, cnts,-1,(255,0,0), 1)
        cv2.imwrite('processed.jpg', img)
        print('finished')
        break
