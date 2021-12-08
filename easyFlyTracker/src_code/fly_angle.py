#! python
# @Time    : 21/05/18 下午 04:31
# @Author  : azzhu 
# @FileName: fly_angle.py
# @Software: PyCharm
import cv2
import math
import cmath
import numpy as np

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
        return self.algorithm3(small_bin_img)

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

    def algorithm3(self, small_bin_img):
        '''
        算法2有致命错误，算的不对，任何情况下都不会输出右下的方向，所以算错了。
        特此更新，使用极坐标的方式计算方向，这回不会错了。
        极坐标返回的角度范围是[-180,180]，如果是负值，会加上180（即相反方向）。
        :param small_bin_img:
        :return:角度，范围[0,180]
        '''
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
        vect_ab = complex(b[0] - a[0], b[1] - a[1])
        cn = cmath.polar(vect_ab)
        ang = math.degrees(cn[1])
        ang = ang if ang >= 0 else ang + 180  # 0-180
        ang = 180 - ang  # 转换坐标系，从左上原点到左下原点
        return ang

    def isLine01bigthanLine03(self, p0, p1, p3):
        l01 = (p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2
        l03 = (p0[0] - p3[0]) ** 2 + (p0[1] - p3[1]) ** 2
        return l01 >= l03


if __name__ == '__main__':
    cn = cmath.polar(complex(-1, -3 ** 0.5))
    ang = math.degrees(cn[1])
    print(ang)
