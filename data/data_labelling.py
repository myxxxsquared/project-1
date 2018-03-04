from .utils import get_maps
import numpy as np
import cv2

class data_churn(object):
    def __init__(self, thickness=0.2, neighbor=7.0, crop_skel=1.0, *args, **kw):
        """
        initialize an instance
        :param kw: 'data_set': str, 'SynthText', 'totaltext', etc.
                     'start_point','end_point':int, indicating the starting point for the crunching process
               thickness: the thickness of the text center line
               neighbor: the range used for fit the theta
               crop_skel: the length for cropping the text center line (skeleton)
        """
        self.thickness = thickness
        self.neighbor = neighbor
        self.crop_skel = crop_skel
        pass

    def _data_labeling(self, img_name, img, cnts, is_text_cnts, left_top, right_bottom, chars, care):
        '''
        :param img_name: pass to return directly, (to be determined, int or str)
        :param img: ndarray, np.uint8,
        :param cnts:
                if is_text_cnts is True: list(ndarray), ndarray: dtype np.float32, shape [n, 1, 2], order(col, row)
                if is_text_cnts is False: list(list(ndarray), list(ndarray)), for [char_cnts, text_cnts]
        :param is_text_cnts: bool
        :param left_top: for cropping
        :param right_bottom: for cropping
        :param chars:
                if is_text_cnts is True: None
                if is_text_cnts is False: a nested list storing the chars info for synthtext
        :return:
                img_name: passed down
                img: np.ndarray np.uint8
                maps: label map with a size of (512, 512, 5) aranged as [TCL, radius, cos_theta, sin_theta, TR], all of them are 2-d array,
                TR: np.bool; TCL: np.bool; radius: np.float32; cos_theta/sin_theta: np.float32
        '''
        try:
            #t = time.time()
            skels_points, radius_dict, score_dict, cos_theta_dict, sin_theta_dict, mask_fills = \
                get_maps(img, cnts, is_text_cnts, self.thickness, self.crop_skel, self.neighbor, char)
            #print(img_name,'label takes:', time.time() - t)
            #t = time.time()
            TR = mask_fills[0]
            for i in range(1, len(mask_fills)):
                TR = np.bitwise_or(TR, mask_fills[i])
            TCL = np.zeros(img.shape[:2], np.bool)
            for point, _ in score_dict.items():
                TCL[point[0], point[1]] = True
            radius = np.zeros(img.shape[:2], np.float32)
            for point, r in radius_dict.items():
                radius[point[0], point[1]] = r
            cos_theta = np.zeros(img.shape[:2], np.float32)
            for point, c_t in cos_theta_dict.items():
                cos_theta[point[0], point[1]] = c_t
            sin_theta = np.zeros(img.shape[:2], np.float32)
            for point, s_t in sin_theta_dict.items():
                sin_theta[point[0], point[1]] = s_t
            TR = TR[left_top[0]:right_bottom[0], left_top[1]:right_bottom[1]]
            TCL = TCL[left_top[0]:right_bottom[0], left_top[1]:right_bottom[1]]
            radius = radius[left_top[0]:right_bottom[0], left_top[1]:right_bottom[1]]
            cos_theta = cos_theta[left_top[0]:right_bottom[0], left_top[1]:right_bottom[1]]
            sin_theta = sin_theta[left_top[0]:right_bottom[0], left_top[1]:right_bottom[1]]
            img = img[left_top[0]:right_bottom[0], left_top[1]:right_bottom[1], :]
            maps = [TCL, radius, cos_theta, sin_theta, TR]
            #print(img_name,'mapping takes:', time.time() - t)
            return img_name, img, np.stack(maps, -1), cnts
        except:
            print('Warning: error encountered in %s' % img_name)
            return None, None, None, None

    def _pixellink_labeling(self, img_name, img, cnts, is_text_cnts, left_top, right_bottom, chars, care):

        cv2.imwrite('img.jpg', img)
        cv2.imwrite('cnts.jpg', cv2.drawContours(img, cnts, -1,255,1))
        # mask
        mask = cv2.drawContours(np.zeros(img.shape[:2]), cnts, -1, 255, 1)
        mask = np.sign(mask).astype(np.float32)

        # links
        links = [np.zeros(img.shape[:2], np.float32) for _ in range(8)]
        '''
        0 1 2
        3 x 4
        5 6 7
        '''
        def _move(cnt, dir):
            '''
            :param cnt: np shape (n, 1, 2) ord: (col, row)
            :param direct:
            :return:
            '''
            directs = [(-1,-1), (0,-1), (1,-1), (-1,0), (1,0), (-1,1),(0,1),(1,1)]
            return cnt+directs[dir]

        for cnt_index in range(len(cnts)):
            base = cv2.drawContours(np.zeros(img.shape[:2]), cnts, cnt_index, 255, 1).astype(np.bool)
            for i in range(8):
                mask_ = cv2.drawContours(np.zeros(img.shape[:2]), [_move(cnts[cnt_index], 7-i)], 0, 255, 1).astype(np.bool)
                links[i][base&mask_] = 1.0

        for i in range(8):
            cv2.imwrite('img'+str(i)+'.jpg', links[i]*255)

        # weight

            # mask_0 = cv2.drawContours(np.zeros(img.shape[:2]), [_move(cnts[cnt_index], 7)], 0, 255, 1)
            # mask_1 = cv2.drawContours(np.zeros(img.shape[:2]), [_move(cnts[cnt_index], 6)], 0, 255, 1)
            # mask_2 = cv2.drawContours(np.zeros(img.shape[:2]), [_move(cnts[cnt_index], 5)], 0, 255, 1)
            # mask_3 = cv2.drawContours(np.zeros(img.shape[:2]), [_move(cnts[cnt_index], 4)], 0, 255, 1)
            # mask_4 = cv2.drawContours(np.zeros(img.shape[:2]), [_move(cnts[cnt_index], 3)], 0, 255, 1)
            # mask_5 = cv2.drawContours(np.zeros(img.shape[:2]), [_move(cnts[cnt_index], 2)], 0, 255, 1)
            # mask_6 = cv2.drawContours(np.zeros(img.shape[:2]), [_move(cnts[cnt_index], 1)], 0, 255, 1)
            # mask_7 = cv2.drawContours(np.zeros(img.shape[:2]), [_move(cnts[cnt_index], 0)], 0, 255, 1)


        maps = 0
        return img_name, img, maps, cnts