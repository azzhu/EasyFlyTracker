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
import time, yaml
import inspect
import ctypes
from pathlib import Path
import pandas as pd
import platform


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


def args_filter(kwargs, fn):
    arg_keys = inspect.getfullargspec(fn).args
    return {k: kwargs[k] for k in kwargs if k in arg_keys}


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
        eps = 1e-4  # 一个比较小的值，加在被除数上，防止除零
        speed = nb / (time_used + eps)  # 速度
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


HELP = \
    '''
Usage: 
    easyFlyTracker [config file path]
    and:
    easyFlyTracker_analysis [config file path]
    
For example:
    easyFlyTracker D:/config.yaml
    
You can find more detail information in this file: config.yaml
    '''


def __get_params():
    args = sys.argv
    if len(args) == 1:
        print(HELP)
        exit()
    if args[1] == '-h' or args[1] == '--help':
        print(HELP)
        exit()
    cfg_p = Path(args[1])
    if not cfg_p.exists():
        print('please check that if the config file path is correct')
        exit()
    with open(cfg_p, 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f)
    vp = params['video_path']
    if vp is None or not Path(vp).exists():
        print('The [video_path] is not existing, please check it!')
        exit()
    return params


def __load_group(params):
    p = params['group_path']
    groups = []
    if p:  # 配置了该路径
        if Path(p).exists():
            try:
                df = pd.read_excel(p, engine='openpyxl')
            except:  # 如果这里面的还不能正常执行，那就报错吧
                df = pd.read_excel(p)
            vs = df.values
            cs = df.columns.values

            for i, c in enumerate(cs):
                flag = c
                gp = [int(v) for v in vs[:, i] if not np.isnan(v)]
                gp = sorted(list(set(gp)))
                groups.append([gp, flag])

            if len(groups) == 0:  # 空文件
                return [[None, 'all']]
            else:
                return groups
        else:
            print('the path, [group_path], is not exists!')
            exit()
    else:  # 未配置该路径
        return [[None, 'all']]  # roi设为空的时候就是全部，色即是空，空即是色。


def gen_reqs():
    import os
    os.system('pipreqs ./ --encoding utf-8')


if __name__ == '__main__':
    def fn(k):
        return 100 / k


    try:
        res = fn(0)
    except:
        res = fn(0)
    print(res)
