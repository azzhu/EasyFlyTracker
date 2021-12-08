#!/GPFS/zhangli_lab_permanent/zhuqingjie/env/py3_tf2/bin/python
'''
@Time    : 20/10/14 上午 09:37
@Author  : zhuqingjie 
@User    : zhu
@FileName: Camera_Calibration.py
@Software: PyCharm
'''
'''
摄像机标定

问题：
使用摄像头拍摄较密集的圆盘时（本实验8*9），离中心区域越远的圆盘畸变越大，导致影响实验结果。

目的：
使用摄像机标定可以很大的改善拍摄畸变的问题，可以使实验计算结果更加准确。

摄像机标定步骤：
1，打印出标定图像（标准棋盘格子，正方形），放在板子上各种方位拍摄若干张（可以使用截图软件把镜头内容截下来）；
2，使用这些图像计算出畸变修正参数；
3，使用畸变修正参数来修正视频结果。

注意：
保存的图像尺寸最好跟视频尺寸一致，如不一致最好进行crop；
同一个摄像头，类似环境，可以使用相同的畸变修正参数。即，同一拍摄环境只需一次采集标定图像，一次计算修正参数，然后后续都可以使用该参数。
'''
from pathlib import Path

import cv2
import numpy as np

from easyFlyTracker.src_code.utils import Wait


def calibration(imgfilelist, mapsavedpath, chess_size=(6, 9)):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chess_size[0] * chess_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chess_size[0], 0:chess_size[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    for fname in imgfilelist:
        img = cv2.imread(fname)
        # img = img[:720]  # 由于截的图像把下面的状态栏也截了，所以这里把状态栏的部分去掉，使跟视频图像尺寸保持一致   #######################
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, chess_size, None)
        # If found, add object points, image points (after refining them)
        # print(ret, fname)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            # img = cv2.drawChessboardCorners(img, chess_size, corners2, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(30)
    # cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    h, w = gray.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    np.save(mapsavedpath, [mapx, mapy])


# 给cli用的，加入了异常判断
def cam_calibration(params):
    img_dir = Path(params['calibration_images_files_dir'])
    mapsavedpath = Path(params['calibration_model_saved_dir'])
    chess_size = params['chess_size']
    if not img_dir.exists():
        print('path [calibration_images_files_dir] is not exists, please check it!')
        exit()
    if not mapsavedpath.exists():
        print('path [calibration_model_saved_dir] is not exists, please check it!')
        exit()
    if type(chess_size) is not list or \
            len(chess_size) != 2 or \
            type(chess_size[0]) is not int or \
            type(chess_size[1]) is not int or \
            chess_size[0] < 2 or \
            chess_size[1] < 2:
        print('param [chess_size] must be a list of two int values, eg:[6,9], please check it!')
        exit()
    mapsavedpath = Path(mapsavedpath, 'model.npy')

    imgfilelist = list(img_dir.iterdir())
    imgfilelist = [str(f) for f in imgfilelist if f.suffix[1:] in ['tif', 'tiff', 'png', 'bmp', 'jpg', 'jpeg']]

    with Wait('Camera Calibration'):
        # 开始相机标定参数计算
        calibration(imgfilelist, mapsavedpath, (chess_size[0], chess_size[1]))


class Undistortion():
    def __init__(self, mapxy_path):
        self.notdo = False
        if mapxy_path == None:
            self.notdo = True
        else:
            self.mapxy = np.load(mapxy_path)

    def do(self, img):
        if self.notdo: return img
        # cv2.imshow('0', img)
        img = cv2.remap(img, *self.mapxy, cv2.INTER_LINEAR)
        # cv2.imshow('1', img)
        # cv2.waitKey()
        return img


if __name__ == '__main__':
    # datas_dir = 'Z:/dataset/qususu/Camera_Calibration/'
    mapxy_path = r'Z:\dataset\qususu\mapx_y.npy'
    # images = [f'{datas_dir}traindatas/{i}.bmp' for i in range(1, 23)]
    # calibration(images, mapxy_path)

    _, img = cv2.VideoCapture(r'Z:\dataset\qususu\1015\202010151010.avi').read()
    ud = Undistortion(r'Z:\dataset\qususu\mapx_y.npy')
    dst = ud.do(img)

    print(img.shape)
    print(dst.shape)
    cv2.imshow('src', img)
    cv2.imshow('dst', dst)
    cv2.waitKey()
