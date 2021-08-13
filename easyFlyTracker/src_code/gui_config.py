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
import math, time
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
    
    21.02.19
    修改所有圆环统一半径
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
        self.init_res = init_res  # 没有config.pkl时用，有了就用不到它了
        self.winname = 'gui config'
        self.lbutton_down = False

        if len(init_res) == 0:
            self.init_res = [[int(w / 2), int(h / 2), int(min(h, w) / 20)]]
        self.roi_id = 0  # 当前操作的圆环

        self.tim = 0  # 时间

        # result
        self.res = []
        self._init_res_2_res()
        self.AB_ps = []

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
                # self.res[self.roi_id][2] += 1
                self.r += 1
            else:  # 下滚
                # self.res[self.roi_id][2] -= 1
                self.r -= 1

    def _opencv_mouse_callback_new(self, event, x, y, flags, param):
        '''
        跟非new的区别：有时候mac系统很煞笔，识别不出左键和右键的双击动作，只能识别出单击，
        那好，我就根据两次单击的时间差来判定是否是双击。
        :return:
        '''
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键按下【设置感兴趣圆环】
            self.lbutton_down = True
            self._get_point_in_which_circle(x, y)

        elif event == cv2.EVENT_LBUTTONUP:  # 左键松开
            self.lbutton_down = False
            # 极短时间内两次松开，即判定为双击
            if time.perf_counter() - self.tim < 0.3:  # # 左键双击【在双击的地方添加新的圆环】
                self._res_add(x, y)
            self.tim = time.perf_counter()

        elif self.lbutton_down and event == cv2.EVENT_MOUSEMOVE:  # 左键按下并且鼠标移动【移动感兴趣圆环】
            # 先看一下点击的点离roi_id的距离
            x0, y0 = self.res[self.roi_id][:2]
            dist = math.sqrt(pow(x - x0, 2) + pow(y - y0, 2))
            # 根据距离判断点击的点是否在圈内
            if dist <= self.res[self.roi_id][2]:
                self.res[self.roi_id][:2] = [x, y]

        elif event == cv2.EVENT_RBUTTONUP:  # 右键松开
            # 极短时间内两次松开，即判定为双击
            if time.perf_counter() - self.tim < 0.3:  # # 右键双击【删除圆环】
                self._get_point_in_which_circle(x, y)
                if len(self.res) > 0: del self.res[self.roi_id]
            self.tim = time.perf_counter()

        elif event == cv2.EVENT_MOUSEWHEEL:  # 上下滚轮【控制直径大小】
            if flags > 0:  # 上滚
                # self.res[self.roi_id][2] += 1
                self.r += 1
            else:  # 下滚
                # self.res[self.roi_id][2] -= 1
                self.r -= 1

    def _init_res_2_res(self):
        if Path(self.res_pkl).exists():
            self.res, self.AB_dist = pickle.load(open(self.res_pkl, 'rb'))
            print(f'config params load from: {self.res_pkl}')
        else:
            self.res = self.init_res
        # 统一的半径
        self.r = int(round(np.mean(np.array(self.res)[:, -1])))

    def _res_add(self, x, y):
        r1 = self.res[self.roi_id][-1] if len(self.res) > 0 else 50
        self.res.append([x, y, r1])
        self.roi_id = len(self.res) - 1

    def _draw_img_from_res(self, skip_ABline=False):
        '''
        :param skip_ABline:
            为什么要skip画AB线呢？因为config.pkl里面只有AB的长度信息，没有AB的坐标信息，
            所以当跳过配置，只从config.pkl文件里面读的时候没有AB坐标信息，就画不了。
        :return:
        '''
        img = self.img.copy()

        # 修改所有圆环半径，保持一致
        for i in range(len(self.res)):
            self.res[i][2] = self.r

        # 画圆
        for i, (x, y, r) in enumerate(self.res):
            cv2.circle(img, (x, y), r, self.col, 1)
            cv2.putText(img, f'{i}', (x - 5, y + 5), 1, 1, self.col)

        # 画AB两条线
        if not skip_ABline:
            ab = ['A', 'B']
            for i, p in enumerate(self.AB_ps):
                cv2.circle(img, p, 3, (0, 0, 220), -1)
                cv2.putText(img, ab[i], (p[0] + 5, p[1] - 5), 0, 0.7, (220, 220, 0), 2)
            cv2.line(img, *self.AB_ps, (0, 0, 220), 1)

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

    def CFG_sacle(self):
        '''
        第一步，配置比例尺
        :return:
        '''
        print('Please set the two points (A and B).')

        def draw_img():
            img = self.img.copy()
            for i, p in enumerate(self.AB_ps):
                cv2.circle(img, p, 3, (0, 0, 255), -1)
                cv2.putText(img, ab[i], (p[0] + 5, p[1] - 5), 0, 0.7, (255, 255, 0), 2)
            if len(self.AB_ps) == 2:
                cv2.line(img, *self.AB_ps, (0, 0, 255), 1)
            return img

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONUP:  # 左键松开
                if len(self.AB_ps) < 2:
                    self.AB_ps.append((x, y))

        key_up = (119, 2490368, 126)  # wasd以及右边方向键，同时适配了mac本的方向键
        key_down = (115, 2621440, 125)
        key_left = (97, 2424832, 123)
        key_right = (100, 2555904, 124)
        delete = (3014656, 8)
        h, w = self.img.shape[:2]
        ab = ['A', 'B']
        while True:
            cv2.imshow(self.winname, draw_img())
            cv2.setMouseCallback(self.winname, mouse_callback)
            k = cv2.waitKeyEx(30)
            if len(self.AB_ps) > 0:
                if k in key_up:
                    x, y = self.AB_ps[-1]
                    self.AB_ps[-1] = (x, y - 1 if y - 1 >= 0 else y)
                elif k in key_down:
                    x, y = self.AB_ps[-1]
                    self.AB_ps[-1] = (x, y + 1 if y + 1 < h else y)
                elif k in key_left:
                    x, y = self.AB_ps[-1]
                    self.AB_ps[-1] = (x - 1 if x - 1 >= 0 else x, y)
                elif k in key_right:
                    x, y = self.AB_ps[-1]
                    self.AB_ps[-1] = (x + 1 if x + 1 < w else x, y)
                elif k in delete:
                    del self.AB_ps[-1]
                elif k == 13:  # enter，退出循环
                    if len(self.AB_ps) != 2:
                        print(f'Please set two points (A and B)!')
                    else:
                        # cv2.destroyWindow(self.winname)
                        break
        # 根据勾股定理计算AB之间的距离
        a, b = self.AB_ps[0], self.AB_ps[1]
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def CFG_circle(self, direct_get_res=False):
        if direct_get_res:
            # 直接退出之前还是得保存config.bmp给用户看
            self._draw_img_from_res(skip_ABline=True)
            cv2_ext.imwrite(str(self.res_pkl)[:-3] + 'bmp', self.drawed_img)
            return self.res, self.AB_dist

        # 先进行第一步：配置比例尺，配置AB两点
        AB_dist = self.CFG_sacle()

        # 后面再进行第二步，配置圆环

        # 键盘右边上下左右键waitKeyEx返回值，有多个值可能是以下原因：需要多个键都能触发、windows和mac键值不一样
        key_up = (119, 2490368, 126)  # wasd以及右边方向键，同时适配了mac本的方向键
        key_down = (115, 2621440, 125)
        key_left = (97, 2424832, 123)
        key_right = (100, 2555904, 124)
        delete = (3014656, 8)
        enlarge = (122, 61)  # 'z' '='
        shrink = (120, 45)  # 'x' '-'

        while True:
            # cv2.imshow('img', self.img)
            cv2.imshow(self.winname, self._draw_img_from_res())
            cv2.setMouseCallback(self.winname, self._opencv_mouse_callback_new)
            k = cv2.waitKeyEx(5)

            # 正常移动
            if k in key_up:  # w，向上移动
                self.res[self.roi_id][1] -= 1
            elif k in key_down:  # s，向下
                self.res[self.roi_id][1] += 1
            elif k in key_left:  # a，向左
                self.res[self.roi_id][0] -= 1
            elif k in key_right:  # d，向右
                self.res[self.roi_id][0] += 1
            # 按住shift再按相应按键【整体移动】
            elif k == key_up:  # w，整体向上移动
                for i in range(len(self.res)): self.res[i][1] -= 1
            elif k == key_down:  # s，整体向下
                for i in range(len(self.res)): self.res[i][1] += 1
            elif k == key_left:  # a，整体向左
                for i in range(len(self.res)): self.res[i][0] -= 1
            elif k == key_right:  # d，整体向右
                for i in range(len(self.res)): self.res[i][0] += 1

            elif k in enlarge:  # z，增加半径
                # self.res[self.roi_id][2] += 1
                self.r += 1
            elif k in shrink:  # x，减少半径
                # self.res[self.roi_id][2] -= 1
                self.r -= 1

            elif k in delete:  # 删除圆环
                if len(self.res) > 0: del self.res[self.roi_id]

            elif k == 13:  # enter，退出循环
                cv2.destroyWindow(self.winname)
                break

        # save and return
        res = (self.res, AB_dist)
        with open(self.res_pkl, 'wb') as f:
            pickle.dump(res, f)
            print(f'saved config params to: {self.res_pkl}')
        cv2_ext.imwrite(str(self.res_pkl)[:-3] + 'bmp', self.drawed_img)
        return res


if __name__ == '__main__':
    # 从左到右从上到下配置圆环
    frame = cv2.imread(r'Z:\dataset\qususu\202108100950_205771.tif')
    # frame = np.ones([500, 700, 3], np.uint8) * 255
    g = GUI_CFG(frame, [], r'Z:\dataset\qususu')
    g.CFG_circle()
    # exit()

    # 这个配置的结果转为之前程序可以使用的
    dir = Path(r'Z:\dataset\qususu')
    pkl_p = Path(dir, 'config.pkl')

    with open(pkl_p, 'rb') as f:
        res = pickle.load(f)

    da = np.array(res[0])
    np.save(Path(dir, 'all_centre_points.npy'), da)
    with open(Path(dir, 'all_centre_points.pkl'), 'wb') as f:
        pickle.dump(da, f)
