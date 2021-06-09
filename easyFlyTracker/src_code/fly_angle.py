#! python
# @Time    : 21/05/18 下午 04:31
# @Author  : azzhu 
# @FileName: fly_angle.py
# @Software: PyCharm
import numpy as np
import cv2
import math

'''
该模块主要用来计算果蝇的角度朝向，基于分割后的二值图像来计算。

但正真的方向可能是相反方向，需要后期再根据其他因素（比如速度方向）来最终确定到底是哪个（二选一）。

返回的是角度，不是弧度，以左下为坐标原点的角度数。

用法示例见main函数。
'''


class Fly_angle:
    def __init__(self):
        self.outlier = 0

    def __call__(self, small_bin_img):
        '''
        建议传入的是一个很小的局部二值图像，因为算的时候会取最大轮廓的那个来算
        :param small_bin_img:
        :return:
        '''
        return self.algorithm2(small_bin_img)

    def algorithm1(self, small_bin_img):
        contours, hierarchy = cv2.findContours(small_bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:  # 没有发现轮廓，返回异常值
            return self.outlier
        max_id = np.argmax(np.array([len(_) for _ in contours]))  # 最大轮廓id
        rec = cv2.minAreaRect(contours[max_id])  # 得到的是：（最小外接矩形的中心（x，y），（宽度，高度），旋转角度）
        box = cv2.boxPoints(rec)  # 得到矩阵四个点坐标，该四个点是有顺序的
        if self.isLine01bigthanLine03(box[0], box[1], box[3]):
            ret = 90 - rec[-1]
        else:
            ret = 180 - rec[-1]
        if ret > 180: ret -= 180
        return ret

    def algorithm2(self, small_bin_img):
        contours, hierarchy = cv2.findContours(small_bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:  # 没有发现轮廓，返回异常值
            return self.outlier
        max_id = np.argmax(np.array([len(_) for _ in contours]))  # 最大轮廓id
        rec = cv2.minAreaRect(contours[max_id])  # 得到的是：（最小外接矩形的中心（x，y），（宽度，高度），旋转角度）
        box = cv2.boxPoints(rec)  # 得到矩阵四个点坐标，该四个点是有顺序的
        if self.isLine01bigthanLine03(box[0], box[1], box[3]):
            a, b = box[0], box[1]
        else:
            a, b = box[0], box[3]
        ang = math.acos((a[0] - b[0]) / (((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5))
        ang = ang * 180 / math.pi
        ang = 180 - ang
        return ang

    def isLine01bigthanLine03(self, p0, p1, p3):
        l01 = (p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2
        l03 = (p0[0] - p3[0]) ** 2 + (p0[1] - p3[1]) ** 2
        return l01 >= l03


if __name__ == '__main__':
    from time import time

    da = np.load(r'D:\Pycharm_Projects\qu_holmes_su_release\tests\output\.cache\track_cor.npy')

    fa = Fly_angle()
    for i in range(1, 1000):
        print(f'{i}\t', end='')
        img = cv2.imread(r'C:\Users\33041\Pictures\sp{}.bmp'.format(i), 0)
        if not isinstance(img, np.ndarray):
            exit()
        cv2.imshow('1', img)
        ang = fa(img)
        cv2.putText(img, f'{ang}', (0, 25), 0, 1, 255)
        cv2.imshow('3', img)
        cv2.waitKeyEx()
