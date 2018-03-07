import numpy as np
import cv2
import random
import time


def _pixellink_labeling(img_name, img, cnts, left_top, right_bottom):
    map_shape = (img.shape[0]//2, img.shape[1]//2)
    cnts = [cnt//2 for cnt in cnts]
    cnts = [cnt.astype(np.int32) for cnt in cnts]
    # mask
    mask = cv2.fillPoly(np.zeros(map_shape), cnts, 255)
    mask = np.sign(mask).astype(np.float32)

    # links & weight
    links = [np.zeros(map_shape, np.float32) for _ in range(8)]
    weight = np.ones(map_shape, np.float32)

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

    total_sum = 0
    for cnt_index in range(len(cnts)):
        base = cv2.fillPoly(np.zeros(map_shape), [cnts[cnt_index]], 255).astype(np.bool)
        temp = base > 0
        total_sum+=np.sum(temp)
        weight[temp] = 1/np.sum(temp)/len(cnts)
        # print(1/np.sum(temp)/len(cnts))
        for i in range(8):
            mask_ = cv2.fillPoly(np.zeros(map_shape), [_move(cnts[cnt_index], 7-i)], 255).astype(np.bool)
            links[i][base&mask_] = 1.0
    # weight *= total_sum

    maps = np.stack([mask]+ links+[weight], -1)
    img = img[left_top[0]:right_bottom[0], left_top[1]:right_bottom[1], :]
    left_top = [left_top[0]//2, left_top[1]//2]
    right_bottom = [right_bottom[0]//2,right_bottom[1]//2]
    maps = maps[left_top[0]:right_bottom[0], left_top[1]:right_bottom[1], :]
    # print(maps.shape)
    # print(img.shape)
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

    if random.random() <= 10:
        # counter clock, 0:0, 1: 90; 2: 180, 3: 270,
        rotate_time = random.randint(0,3)
        for i in range(rotate_time):
            img, contour = _rotate(img, contour)
    img = img.copy().astype(np.uint8)

    # step2: resize and aspect
    new_row, new_col=0,0
    row, col = img.shape[:2]
    # t = 0
    incremental = 0.0
    while new_row-512-1 <0 or new_col-512-1 <0:
        size_ratio = random.random()* 2*(1+incremental)
        aspect_ratio = random.random()*1.5+0.5
        new_row, new_col = row*size_ratio, col*size_ratio
        new_row, new_col = new_row, new_col*aspect_ratio
        new_row, new_col = int(new_row), int(new_col)
        incremental += 0.1
        # print(t)
        # t+= 1

    img = cv2.resize(img, (new_col, new_row))
    new = []
    for cnt in contour:
        points = []
        for point in cnt:
            # y is col, x is row
            y, x = point[0]
            new_y, new_x = y*new_col/col, x*new_row/row
            points.append([[new_y, new_x]])
        points = np.array(points, np.float32)
        new.append(points)
    contour = new

    # step3: get crop points
    left_top = random.randint(0,new_row-512-1), random.randint(0,new_col-512-1)
    right_bottom = left_top[0]+512, left_top[1]+512
    return img_name, img, contour, left_top, right_bottom


def pixellink_prepro(ins):
    img_name, img, cnts, maps = _pixellink_labeling(*_pixellink_transform(ins))
    return img_name, img, cnts, maps


def _fit_TCL(img, cnts):
    zeros = np.zeros(img.shape[:2], np.uint8)
    zeros = cv2.fillPoly(zeros, cnts, 255)
    cv2.imwrite('img.jpg', zeros)
    points = np.argwhere(zeros>100)
    # row, col
    # for point in points:
    #     print(point)
    # points = points.transpose()
    # [rows, cols]

    rows,cols = [],[]
    show = np.zeros(img.shape[:2], np.uint8)
    for row,col in points:
        show[row, col] = 255
        rows.append(row)
        cols.append(col)

    # show[points] = 255
    # cv2.imwrite('show.jpg', show)
    #
    # rows,cols = [],[]
    #
    # cnt = cnts[0]
    # for col, row in np.squeeze(cnt):
    #     cols.append(col)
    #     rows.append(row)

    a,b,c,d = np.polyfit(cols,rows,3)
    line_cols = np.linspace(0,500,500)
    line_rows = ((a*line_cols+b)*line_cols+c)*line_cols+d
    line_cols = line_cols.astype(np.int32)
    line_rows = line_rows.astype(np.int32)

    test = np.zeros_like(img, np.uint8)
    test = cv2.fillPoly(test, cnts, 255)
    for row, col in zip(line_rows, line_cols):
        if row <= img.shape[0] and col <= img.shape[1]:
            test[row, col, :] = (255,255,255)
    cv2.imwrite('test.jpg', test)
    return 0


if __name__ == '__main__':
    import os
    import pickle
    TOTAL_TRAIN_DIR = '/home/rjq/data_cleaned/pkl/totaltext_train_care/'
    TOTAL_TEST_DIR = '/Users/ruanjiaqiang/Desktop/totaltext_test/'

    file_names_totaltext_train = [TOTAL_TEST_DIR+name for name in os.listdir(TOTAL_TEST_DIR)]

    for file_name in file_names_totaltext_train:
        ins = pickle.load(open(file_name, 'rb'))
        img = ins['img'].copy()
        cnts = [cnt.astype(np.int32) for cnt in ins['contour']]
        img = cv2.drawContours(img, cnts,-1,(255,0,255), 3)
        cv2.imwrite('origin.jpg', img)
        img_name, img, cnts, maps = pixellink_prepro(ins)

        # img_name, img, cnts, left_top, right_bottom = _pixellink_transform(ins)
        cnts = [cnt.astype(np.int32) for cnt in cnts]
        img = img.astype(np.uint8)
        img = cv2.resize(img, (256,256))
        img = cv2.drawContours(img, cnts,-1,(255,0,255), 3)

        for i in range(maps.shape[2]):
            cv2.imwrite('map'+str(i)+'.jpg', maps[:,:,i]*255)
        cv2.imwrite('processed.jpg', img)
        print('finished')
        break

    # img = np.zeros((512,512,3))
    # cnts = [
    #     [
    #         [[100,100]],
    #         [[200,100]],
    #         [[300,300]],
    #         [[300,100]],
    #         [[350,100]],
    #         [[350,400]],
    #         [[250,400]],
    #         [[150,200]],
    #         [[150,400]],
    #         [[100,400]],
    #     ]
    # ]
    #
    # cnts = [
    #     [
    #         [[100,100]],
    #         [[200,100]],
    #         [[300,300]],
    #         [[350,100]],
    #         [[400,100]],
    #         [[350,400]],
    #         [[250,400]],
    #         [[150,200]],
    #         [[100,400]],
    #         [[50,400]],
    #     ]
    # ]
    #
    # cnts = [
    #     [
    #         [[100,100]],
    #         [[200,100]],
    #         [[200,300]],
    #         [[100,300]],
    #     ]
    # ]
    #
    # cnts = [
    #     [
    #         [[100,100]],
    #         [[400,100]],
    #         [[400,200]],
    #         [[100,200]],
    #     ]
    # ]
    #
    # cnts = [
    #     [
    #         [[100,100]],
    #         [[200,200]],
    #         [[300,200]],
    #         [[400,100]],
    #         [[400,200]],
    #         [[300,300]],
    #         [[200,300]],
    #         [[100,200]],
    #     ]
    # ]
    # cnts = [np.array(cnt, np.int32) for cnt in cnts]
    # _fit_TCL(img, cnts)
