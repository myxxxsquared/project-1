import numpy as np
import cv2
from data.utils import get_maps

VIZ_DIR = '/home/rjq/data_cleaned/viz/'


def get_l2_dist(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def _find(Index, Sets):
    for i, Set in enumerate(Sets):
        if Index in Set:
            return i
    return -1


def _intersection(re_mask_list):
    score = {}
    for i in range(len(re_mask_list)):
        for j in range(i, len(re_mask_list)):
            score[(i, j)] = max(np.sum(re_mask_list[i] & re_mask_list[j]) / np.sum(re_mask_list[j]),
                                np.sum(re_mask_list[i] & re_mask_list[j]) / np.sum(re_mask_list[i]))
    return score


def evaluate(img, cnts, is_text_cnts, maps,
             TCL_threshold=0.5, TR_threshold=0.5, fsk=0.8, tp=0.4, tr=0.8, merge_th=0.2):
    '''
    :param img: ndarrray, np.uint8,
    :param cnts:
        if is_text_cnts is True: list(ndarray), ndarray: dtype np.float32, shape [n, 1, 2]
        if is_text_cnts is False: list(list(ndarray), list(ndarray)), for [char_cnts, text_cnts]
    :param is_text_cnts: bool
    :param maps:
        maps: [neg-TCL, TCL, radius, cos_theta, sin_theta, neg-TR, TR], all of them are 2-d array,
        TR: np.bool; TCL: np.bool; radius: np.float32; cos_theta/sin_theta: np.float32
    :param is_viz: bool
    :param save_name
           if is_viz is True, save_name is used to save viz pics
    :return:

    '''
    if not is_text_cnts:
        char_cnts, text_cnts = cnts
        cnts = text_cnts
    print(img.shape[:2], maps[0].shape)
    assert img.shape[:2] == maps[0].shape
    row, col = img.shape[:2]
    [TCL_neg, TCL_pos, radius, cos_theta, sin_theta, TR_neg, TR_pos] = maps
    TR = np.exp(TR_pos) / (np.exp(TR_pos) + np.exp(TR_neg)) > TR_threshold
    TCL = np.exp(TCL_pos) / (np.exp(TCL_pos) + np.exp(TCL_neg)) > TCL_threshold
    # use TR to crop TCL
    cropped_TCL = np.bitwise_and(TR, TCL)

    # pick out instance TCL from cropped_TCL map
    instances = []
    direction_x = [-1, 0, 1]
    direction_y = [-1, 0, 1]

    cropped_TCL_for_search = cropped_TCL.copy()
    while np.sum(cropped_TCL_for_search) != 0:
        instance = []
        queue = []
        xs, ys = np.nonzero(cropped_TCL_for_search)
        queue.append((xs[0], ys[0]))
        cropped_TCL_for_search[xs[0], ys[0]] = False
        while len(queue) != 0:
            x, y = queue.pop(0)
            instance.append((x, y))
            for i in range(len(direction_x)):
                for j in range(len(direction_y)):
                    x_next = x + direction_x[i]
                    y_next = y + direction_y[j]
                    if x_next < row and y_next < col and \
                            cropped_TCL_for_search[x_next, y_next] == True:
                        queue.append((x_next, y_next))
                        cropped_TCL_for_search[x_next, y_next] = False
        instances.append(instance)

    # for each instance build its bounding box(represented by cnt)
    reconstructed_cnts = []
    for instance in instances:
        zeros = np.zeros((row, col), np.uint8)
        for x, y in instance:
            r = radius[x, y]
            # print(r)
            # print(type(r))
            # be careful, in cv2, coordination is (col, row)
            zeros = cv2.circle(zeros, (y, x), r, (255), -1)
        _, cnt, _ = cv2.findContours(zeros, 1, 2)
        if len(cnt) > 1:
            print('more than one cnt')
            for cnt_ in cnt:
                reconstructed_cnts.append(cnt_)
        else:
            reconstructed_cnts.append(cnt)
    reconstructed_cnts = [np.reshape(np.array(reconstructed_cnt, np.float32), (-1, 1, 2))
                          for reconstructed_cnt in reconstructed_cnts]

    viz = np.zeros(img.shape, np.uint8)
    cnts = [np.array(cnt, np.int32) for cnt in cnts]
    viz = cv2.drawContours(viz, cnts, -1, (255, 255, 255), 1)
    reconstructed_cnts = [np.array(cnt, np.int32) for cnt in reconstructed_cnts]
    viz = cv2.drawContours(viz, reconstructed_cnts, -1, (255, 0, 0), 1)

    cnts_num = len(cnts)
    re_cnts_num = len(reconstructed_cnts)

    cnts_mask = []
    re_cnts_mask = []

    for i in range(cnts_num):
        zeros = np.zeros(img.shape[:2], np.uint8)
        cnts_mask.append(cv2.fillPoly(zeros, [cnts[i]], (255)).astype(np.bool))
    for i in range(re_cnts_num):
        zeros = np.zeros(img.shape[:2], np.uint8)
        re_cnts_mask.append(cv2.fillPoly(zeros, [reconstructed_cnts[i]], (255)).astype(np.bool))

    # merge highly-intersected text instances for i/min(area) > 0.2

    intersection_score = _intersection(re_cnts_mask)
    intersection_set = []
    #print(intersection_score)
    for i in range(len(re_cnts_mask)):
        for j in range(i, len(re_cnts_mask)):
            if intersection_score[(i, j)] > merge_th:
                i1 = _find(i, intersection_set)
                i2 = _find(j, intersection_set)
                if i1 < 0 and i2 < 0:
                    intersection_set.append({i, j})
                elif i1 >= 0 and i2 < 0:
                    intersection_set[i1].add(j)
                elif i2 >= 0 and i1 < 0:
                    intersection_set[i2].add(i)
                elif i1 != i2:
                    intersection_set[i1] = intersection_set[i1] | intersection_set[i2]
                    intersection_set.pop(i2)
            #print(intersection_set)
            #x = input('next')

    merged_re_cnts_mask = []
    new_cnts = []

    for Set in intersection_set:
        zeros = np.zeros(img.shape[:2], np.uint8).astype(np.bool)
        for Index in Set:
            zeros = zeros | re_cnts_mask[Index]
        merged_re_cnts_mask.append(zeros)
        #cv2.imwrite(filename='trial'+'_%d.jpg'%(list(Set)[0]),img=zeros*255)
        _, cnt, _ = cv2.findContours(zeros.astype(np.uint8), 1, 2)
        if len(cnt) > 1:
            print('more than one cnt')
            for cnt_ in cnt:
                new_cnts.append(cnt_)
        else:
            new_cnts.append(cnt)
    re_cnts_num = len(merged_re_cnts_mask)
    new_cnts = [np.reshape(np.array(new_cnt, np.float32), (-1, 1, 2))
                for new_cnt in new_cnts]
    new_cnts = [np.array(cnt, np.int32) for cnt in new_cnts]
    viz = cv2.drawContours(viz, new_cnts, -1, (255, 255, 0), 1)

    precise = np.zeros((cnts_num, re_cnts_num), np.float32)
    for i in range(cnts_num):
        for j in range(re_cnts_num):
            precise[i, j] = np.sum(cnts_mask[i] & merged_re_cnts_mask[j]) / np.sum(merged_re_cnts_mask[j])

    recall = np.zeros((cnts_num, re_cnts_num), np.float32)
    for i in range(cnts_num):
        for j in range(re_cnts_num):
            recall[i, j] = np.sum(cnts_mask[i] & merged_re_cnts_mask[j]) / np.sum(cnts_mask[i])

    IOU = np.zeros((cnts_num, re_cnts_num), np.float32)
    for i in range(cnts_num):
        for j in range(re_cnts_num):
            IOU[i, j] = np.sum(cnts_mask[i] & merged_re_cnts_mask[j]) / \
                np.sum(cnts_mask[i] | merged_re_cnts_mask[j])


    gt_score = np.zeros((cnts_num), np.float32)
    pred_score = np.zeros((re_cnts_num), np.float32)
    flag_gt = np.zeros((cnts_num), np.int32)
    flag_pred = np.zeros((re_cnts_num), np.int32)

    # one to one
    for i in range(cnts_num):
        match_r_num = np.sum(recall[i,:]>=tr)
        match_p_num = np.sum(precise[i,:]>=tp)
        if match_p_num==1 and match_r_num==1:
            gt_score[i] = 1.0
            flag_gt[i] = 1
            j = int(np.argwhere(precise[i,:]>=tp))
            j_ = int(np.argwhere(recall[i,:]>=tr))
            if j != j_:
                print('i',i,'j',j,'j_', j_)
                print('precise', precise)
                print('recall', recall)
            pred_score[j] = 1.0
            flag_pred[j] = 1

    # one to many
    for i in range(cnts_num):
        if flag_gt[i] >0:
            continue
        index_list = []
        for j in range(re_cnts_num):
            if precise[i,j] >= tp and flag_pred[j] == 0:
                index_list.append(j)
        r_sum = 0.0
        for j in index_list:
            r_sum += recall[i,j]
        if r_sum >= tr:
            assert len(index_list) > 1
            gt_score[i] = fsk
            flag_gt[i] = 1
            for j in index_list:
                pred_score[j] = fsk
                flag_pred[j] = 1

    # many to one
    for j in range(re_cnts_num):
        if flag_pred[j] >0:
            continue
        index_list = []
        for i in range(cnts_num):
            if recall[i,j] >= tr and flag_gt[i] == 0:
                index_list.append(i)
        p_sum = 0.0
        for i in index_list:
            p_sum += precise[i,j]
        if p_sum >= tp:
            assert len(index_list) > 1
            pred_score[j] = fsk
            flag_pred[j] = 1
            for i in index_list:
                gt_score[i] = fsk
                flag_gt[i] = 1



    totaltext_recall = np.sum(gt_score) / cnts_num
    totaltext_precision = np.sum(pred_score) / re_cnts_num

    pascal_gt_score = np.zeros((cnts_num), np.float32)
    pascal_pred_score = np.zeros((re_cnts_num), np.float32)
    for i in range(cnts_num):
        for j in range(re_cnts_num):
            if IOU[i, j] >= 0.5:
                if pascal_gt_score[i] == 1.0:
                    pascal_pred_score[j] = 0.0
                else:
                    pascal_pred_score[j] = 1.0
                pascal_gt_score[i] = 1.0
    pascal_recall = np.sum(pascal_gt_score) / cnts_num
    pascal_precision = np.sum(pascal_pred_score) / re_cnts_num

    return totaltext_recall, totaltext_precision, \
        pascal_recall, pascal_precision


if __name__ == '__main__':
    EVALUATE_DIR = '/home/rjq/data_cleaned/data_cleaned/evaluate/'
    PKL_DIR = '/home/rjq/data_cleaned/pkl/'
    import pickle
    import time

    # ######test char&text cnts##########
    # for i in range(9, 10):
    #     res = pickle.load(open(PKL_DIR+'synthtext/'+str(i)+'.bin', 'rb'))

    # ######test text cnts###############
    # for i in range(99, 100):
    #     res = pickle.load(open(PKL_DIR + 'totaltext_train/' + str(i) + '.bin', 'rb'))

        # print(res['img_name'],
        #       res['contour'],
        #       res['img'],
        #       res['is_text_cnts'])

        # img_name = res['img_name']
        # img_name = img_name.replace('/', '_')
        # img = res['img']
        # cnts = res['contour']
        # is_text_cnts = res['is_text_cnts']
        #
        # skels_points, radius_dict, score_dict, cos_theta_dict, sin_theta_dict, mask_fills = \
        #     get_maps(img, cnts, is_text_cnts, 0.15, 1.0, 2)
        # TR = mask_fills[0]
        # for i in range(1, len(mask_fills)):
        #     TR = np.bitwise_or(TR, mask_fills[i])
        # TCL = np.zeros(img.shape[:2], np.bool)
        # for point, _ in score_dict.items():
        #     TCL[point[0], point[1]] = True
        # radius = np.zeros(img.shape[:2], np.float32)
        # for point, r in radius_dict.items():
        #     radius[point[0], point[1]] = r
        # cos_theta = np.zeros(img.shape[:2], np.float32)
        # for point, c_t in cos_theta_dict.items():
        #     cos_theta[point[0], point[1]] = c_t
        # sin_theta = np.zeros(img.shape[:2], np.float32)
        # for point, s_t in sin_theta_dict.items():
        #     sin_theta[point[0], point[1]] = s_t
        #
        # def save_heatmap(save_name, map):
        #     map = np.array(map, np.float32)
        #     if np.max(map) != 0.0 or np.max(map) != 0:
        #         cv2.imwrite(save_name, (map * 255 / np.max(map)).astype(np.uint8))
        #     else:
        #         cv2.imwrite(save_name, map.astype(np.uint8))
        #
        # maps = [TR, TCL, radius, cos_theta, sin_theta]
        # t1 = time.time()
        # totaltext_recall, totaltext_precision, pascal_recall, pascal_precision = \
        #     evaluate(img, cnts, is_text_cnts, maps, True, img_name)
        # t2 = time.time()
        # print(totaltext_recall, totaltext_precision, pascal_recall, pascal_precision,
        #       sep='\n')
        # print('time', t2 - t1)


