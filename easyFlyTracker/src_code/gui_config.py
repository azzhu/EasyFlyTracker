#!/GPFS/zhangli_lab_permanent/zhuqingjie/env/py3_tf2/bin/python
'''
@Time    : 20/08/28 上午 10:46
@Author  : zhuqingjie 
@User    : zhu
@FileName: gui_config.py
@Software: PyCharm
'''
import numpy as np
import cv2, cv2_ext, pickle, random
import math
from pathlib import Path

'''
操作说明：

左键点击：选中圆环；
按住左键并移动：移动圆环；
左键双击：新增圆环；
右键双击：删除圆环；
鼠标滚轮：调整圆环直径大小；

w,s,a,d：微调，上下左右移动；
shift+w,s,a,d：所有圆环整体移动；
z：微调，增加半径；
x：微调，减小半径；

enter：确认调整，并退出。
'''


class GUI_CFG():
    __doc__ = '''
    通过鼠标和键盘来便捷配置一些参数；
    【注意】：返回的结果并没有排序。
    '''

    def __init__(self, img_path, init_res, saved_dir=None):
        '''
        传入两个参数
        :param img_path: 可以是其中一帧的路径，也可以是背景图的路径；
        :param init_res: 类似15*3的一个list，15代表15个圆盘，3代表坐标和半径。启动时默认的结果。
                         可以传空list，不管传什么，只要同文件夹下有config.pkl文件，
                         则读取其结果作为默认config。
        '''
        if type(img_path) is str:
            img = cv2_ext.imread(img_path)
            self.res_pkl = str(Path(img_path).parent) + '/config.pkl'
        else:
            img = img_path
            self.res_pkl = str(saved_dir) + '/config.pkl'
        self.col = (0, 0, 0) if img.mean() > 256 / 2 else (255, 255, 255)
        h, w = img.shape[:2]
        self.srcimg = img.copy()
        self.img = img
        self.drawed_img = img
        self.init_res = init_res
        self.winname = 'gui config'
        self.lbutton_down = False

        if len(init_res) == 0:
            self.init_res = [[int(w / 2), int(h / 2), int(min(h, w) / 20)]]
        self.roi_id = 0  # 当前操作的圆环

        # result
        self.res = []
        self._init_res_2_res()

    def _get_point_in_which_circle(self, x, y):
        get_dist = lambda m, n: math.sqrt(pow(m - x, 2) + pow(n - y, 2))
        for i, v in enumerate(self.res):
            a, b, r = v
            if get_dist(a, b) < r:
                self.roi_id = i
                return

    def _opencv_mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键按下【设置感兴趣圆环】
            self.lbutton_down = True
            self._get_point_in_which_circle(x, y)
        elif event == cv2.EVENT_LBUTTONUP:  # 左键松开
            self.lbutton_down = False
        elif self.lbutton_down and event == cv2.EVENT_MOUSEMOVE:  # 左键按下并且鼠标移动【移动感兴趣圆环】
            # 先看一下点击的点离roi_id的距离
            x0, y0 = self.res[self.roi_id][:2]
            dist = math.sqrt(pow(x - x0, 2) + pow(y - y0, 2))
            # 根据距离判断点击的点是否在圈内
            if dist <= self.res[self.roi_id][2]:
                self.res[self.roi_id][:2] = [x, y]
        elif event == cv2.EVENT_LBUTTONDBLCLK:  # 左键双击【在双击的地方添加新的圆环】
            self._res_add(x, y)
        elif event == cv2.EVENT_RBUTTONDBLCLK:  # 右键双击【删除圆环】
            self._get_point_in_which_circle(x, y)
            if len(self.res) > 0: del self.res[self.roi_id]
        elif event == cv2.EVENT_MOUSEWHEEL:  # 上下滚轮【控制直径大小】
            if flags > 0:  # 上滚
                self.res[self.roi_id][2] += 1
            else:  # 下滚
                self.res[self.roi_id][2] -= 1

    def _init_res_2_res(self):
        if Path(self.res_pkl).exists():
            self.res = pickle.load(open(self.res_pkl, 'rb'))
            print(f'config params load from: {self.res_pkl}')
        else:
            self.res = self.init_res

    def _res_add(self, x, y):
        r1 = self.res[self.roi_id][-1] if len(self.res) > 0 else 50
        self.res.append([x, y, r1])
        self.roi_id = len(self.res) - 1

    def _draw_img_from_res(self):
        img = self.img.copy()

        # 画圆
        for i, (x, y, r) in enumerate(self.res):
            cv2.circle(img, (x, y), r, self.col, 1)
            cv2.putText(img, f'{i}', (x - 5, y + 5), 1, 1, self.col)

        # 越界判断
        if self.roi_id < 0: self.roi_id = 0
        if self.roi_id >= len(self.res): self.roi_id = len(self.res) - 1

        if len(self.res) > 0:
            x, y, r = self.res[self.roi_id]
            cv2.circle(img, (x, y), r, (0, 0, 255), 1)
            cv2.putText(img, f'{self.roi_id}', (x - 5, y + 5), 1, 1, (0, 0, 255), 2)

        # 当前画的图像备份，退出的时候保存时用
        self.drawed_img = img
        return img

    def CFG_circle(self, direct_get_res=False):
        if direct_get_res:
            # 直接退出之前还是得保存config.bmp给用户看
            self._draw_img_from_res()
            cv2_ext.imwrite(str(self.res_pkl)[:-3] + 'bmp', self.drawed_img)
            return self.res

        # 键盘右边上下左右键waitKeyEx返回值
        key_up, key_down, key_left, key_right = (
            2490368,
            2621440,
            2424832,
            2555904)

        while True:
            # cv2.imshow('img', self.img)
            cv2.imshow(self.winname, self._draw_img_from_res())
            cv2.setMouseCallback(self.winname, self._opencv_mouse_callback)
            k = cv2.waitKeyEx(30)

            # 正常移动
            if k == 119 or k == key_up:  # w，向上移动
                self.res[self.roi_id][1] -= 1
            elif k == 115 or k == key_down:  # s，向下
                self.res[self.roi_id][1] += 1
            elif k == 97 or k == key_left:  # a，向左
                self.res[self.roi_id][0] -= 1
            elif k == 100 or k == key_right:  # d，向右
                self.res[self.roi_id][0] += 1
            # 按住shift再按相应按键【整体移动】
            elif k == 87:  # w，整体向上移动
                for i in range(len(self.res)): self.res[i][1] -= 1
            elif k == 83:  # s，整体向下
                for i in range(len(self.res)): self.res[i][1] += 1
            elif k == 65:  # a，整体向左
                for i in range(len(self.res)): self.res[i][0] -= 1
            elif k == 68:  # d，整体向右
                for i in range(len(self.res)): self.res[i][0] += 1

            elif k == 122:  # z，增加半径
                self.res[self.roi_id][2] += 1
            elif k == 120:  # x，减少半径
                self.res[self.roi_id][2] -= 1
            elif k == 13:  # enter，退出循环
                cv2.destroyWindow(self.winname)
                break

        # save and return
        with open(self.res_pkl, 'wb') as f:
            pickle.dump(self.res, f)
            print(f'saved config params to: {self.res_pkl}')
        cv2_ext.imwrite(str(self.res_pkl)[:-3] + 'bmp', self.drawed_img)
        return self.res


if __name__ == '__main__':
    p_ = r'D:\Pycharm_Projects\qu_holmes_su_release\tests\qudashen.mp4'
    cap = cv2.VideoCapture(p_)
    _, frame = cap.read()

    g = GUI_CFG(frame, [], r'D:\Pycharm_Projects\qu_holmes_su_release\tests')
    res = g.CFG_circle()
    exit()
