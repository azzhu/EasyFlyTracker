#!/GPFS/zhangli_lab_permanent/zhuqingjie/env/py3_tf2/bin/python
'''
@Time    : 20/07/01 下午 04:06
@Author  : zhuqingjie 
@User    : zhu
@FileName: utils.py
@Software: PyCharm
'''
import sys, os
import numpy as np
import threading
import time
import inspect
import ctypes
from pathlib import Path


def stop_thread(thread):
    """raises the exception, performs cleanup if needed"""
    tid, exctype = thread.ident, SystemExit
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def printc(s='null', **kargs):
    '''
    亮瞎眼的print
    :param s:
    :return:
    '''
    print(f"\033[1;31m{s}\033[0m", **kargs)


def get_Class_params(cfg, cls):
    '''
    从cfg文件获取flyseg的传参字典
    当flyseg参数有变化的时候这个地方也有对应变化

    只返回从cfg中读取的参数，某些参数（如roi_flys_mask_arry）若需要修改，可以在返回结果中从新修改
    :param cfg:
    :return:
    '''
    if cls.__doc__ == 'flyseg':
        raise NotImplementedError
    elif cls.__doc__ == 'ana':
        args = ['video_path', 'h_num', 'w_num', 'roi_flys_mask_arry', 'sleep_time_th',
                'area_th', 'duration_time', 'ana_time_duration',
                'minR_maxR_minD', 'sleep_dist_th_per_second', 'dish_exclude',
                ]
        params = {arg: cfg[arg] for arg in args}
        params.update({
            # 'dish_exclude': None,
        })
        return params


class NumpyArrayHasNanValuesExceptin(Exception):
    '''
    numpy矩阵含有nan数值的异常
    '''

    def __init__(self, arr):
        self.arr = arr

    def __str__(self):
        ret = '\nNumpyArrayHasNanValuesExceptin:\n' + \
              f'shape:\t\t{self.arr.shape}\n' + \
              f'min:\t\t{self.arr.min()}\n' + \
              f'max:\t\t{self.arr.max()}\n' + \
              f'nan num:\t{np.isnan(self.arr).sum()}'
        return ret


class Pbar():
    '''
    煞笔tqdm不好用，自己实现一个进度条，使用方法类似tqdm。
    特点：
    不会出现一直换行问题，只保留一行更新。
    格式：
    【完成百分比 进度条 当前任务/总共任务 [总耗时<剩余时间，速度]】
    '''

    def __init__(self, total, pbar_len=50, pbar_value='|', pbar_blank='-'):
        self.total = total
        self.pbar_len = pbar_len
        self.pbar_value = pbar_value
        self.pbar_blank = pbar_blank
        self.now = 0
        self.time = time.time()
        self.start_time = time.time()

    def update(self, nb=1, set=False, set_value=None):
        if set:
            self.now = set_value
        else:
            self.now += nb
        percent = round(self.now / self.total * 100)  # 百分比数值
        pbar_now = round(self.pbar_len * percent / 100)  # 进度条当前长度
        if pbar_now > self.pbar_len: pbar_now = self.pbar_len  # 允许now>total，但是不允许pbar_now>pbar_len
        blank_len = self.pbar_len - pbar_now
        time_used = time.time() - self.time  # 当前更新耗时
        speed = nb / time_used  # 速度
        total_time_used = time.time() - self.start_time  # 总耗时
        total_time_used_min, total_time_used_sec = divmod(total_time_used, 60)
        total_time_used = f'{int(total_time_used_min):0>2d}:{int(total_time_used_sec):0>2d}'
        remaining_it = self.total - self.now if self.total - self.now >= 0 else 0  # 剩余任务
        remaining_time = remaining_it / speed  # 剩余时间
        remaining_time_min, remaining_time_sec = divmod(remaining_time, 60)
        remaining_time = f'{int(remaining_time_min):0>2d}:{int(remaining_time_sec):0>2d}'
        pbar = f'{percent:>3d}%|{self.pbar_value * pbar_now}{self.pbar_blank * blank_len}| ' \
               f'{self.now}/{self.total} [{total_time_used}<{remaining_time}, {speed:.2f}it/s]'
        print(f'\r{pbar}', end='')
        self.time = time.time()

    def close(self):
        print()  # 把光标移到下一行


def del_files(video_path):
    # 删除特定文件
    video_path = Path(video_path)
    dir = video_path.parent
    stem = video_path.stem

    files_list = [
        '*.bmp',
        f'show_result_{stem}/',
        f'{stem}/',
        # f'{stem}/all_cen*',
        # f'{stem}/all_fly*',
        # f'{stem}/total.npy',
        # f'{stem}/analysis_result/',
    ]

    for f in files_list:
        cmd = f'/bin/rm -rf {dir}/{f}'
        print(cmd)
        os.system(cmd)


def gen_reqs():
    import os
    os.system('pipreqs ./ --encoding utf-8')


if __name__ == '__main__':
    gen_reqs()
    exit()
    # from analysis import Analysis
    # from load_configyaml import load_config
    #
    # cf = load_config()
    # params = get_Class_params(cf, Analysis)
    # print(params)
    # del_files('/home/zhangli_lab/zhuqingjie/dataset/qususu/1012/202010121025.avi')
    import cv2
    from Camera_Calibration import Undistortion

    img0923 = cv2.imread(r'Z:\dataset\qususu\0923\202009231045_bg.bmp')
    img1012 = cv2.imread(r'Z:\dataset\qususu\1012\202010121025_bg.bmp')
    ud = Undistortion(r'Z:\dataset\qususu\mapx_y.npy')
    img0923 = ud.do(img0923, sc=31 / 46)
    img1012 = ud.do(img1012, sc=1)
    cv2.imshow('1012', img1012)
    cv2.imshow('0923', img0923)
    cv2.waitKey()

    pb = Pbar(total=50)
    for i in range(50):
        time.sleep(0.1)
        _, sv = divmod(i, 10)
        pb.update(set=True, set_value=sv)
    pb.close()
    print('ok')
