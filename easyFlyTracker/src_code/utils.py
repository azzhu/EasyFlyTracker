#!/GPFS/zhangli_lab_permanent/zhuqingjie/env/py3_tf2/bin/python
'''
@Time    : 20/07/01 下午 04:06
@Author  : zhuqingjie 
@User    : zhu
@FileName: utils.py
@Software: PyCharm
'''
import ctypes
import inspect
import sys
import time
from collections import Counter
from multiprocessing import Process
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from easyFlyTracker import __version__ as version


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


def equalizeHist_use_mask(gray, mask, notuint8=False):
    if notuint8:
        mv = gray.max()
    else:
        mv = 255
    h, w = gray.shape
    mask = mask != 0
    roi_num = mask.sum()  # roi区域像素个数
    bg_num = h * w - roi_num  # 背景区域像素个数
    gray *= mask
    dc = Counter(gray.flatten())
    count = np.array([dc.get(i, 0) for i in range(mv + 1)], np.float)
    count[0] -= bg_num  # 减去背景像素个数
    count = count / roi_num  # 得到概率
    # 计算累计概率
    for i in range(1, mv + 1):
        count[i] += count[i - 1]
    # 映射
    map1 = count * mv
    ret = map1[gray]
    # 映射之后最小值不是零，所以再做一次归一化，变为0-mv
    min_v = map1.min()
    max_v = map1.max()
    ret = (ret - min_v) / (max_v - min_v) * mv
    if not notuint8:
        ret = np.round(ret).astype(np.uint8)
    return ret


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
        self.close_flag = False

    def update(self, nb=1, set=False, set_value=None):
        if set:
            self.now = set_value
        else:
            self.now += nb
        percent = int(round(self.now / self.total * 100))  # 百分比数值
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

    def close(self, reset_done=True):
        if self.close_flag: return  # 防止多次执行close时会打印多行
        if reset_done:  # 把状态条重置为完成
            self.update(set=True, set_value=self.total)
        print()  # 把光标移到下一行
        self.close_flag = True


class Wait():
    '''
    等待模块。
    跟Pbar类似，Pbar需要更新，需要知道当前进度，这个不需要，只展示了一个等待的动画。
    需要处理的代码块放到with里面即可，注意，该代码块里面最好就不要再有任何print了。

    usage:
    with Wait():
        time.sleep(10)
    '''

    def __init__(self, info=None):
        if info:
            print(f'{info}.../', end='', flush=True)
        else:
            print('.../', end='', flush=True)

    def __enter__(self):
        self.p = Process(target=self.print_fn, args=())
        self.p.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.p.terminate()
        print('\bDone')

    def print_fn(self):
        while True:
            time.sleep(0.3)
            print('\b\\', end='', flush=True)
            time.sleep(0.3)
            print('\b/', end='', flush=True)


HELP = \
    f'''
Version: 
    {version}

Usage: 
    easyFlyTracker [config file path]
    and:
    easyFlyTracker_analysis [config file path]
    
For example:
    easyFlyTracker D:/config.yaml
    
You can find more detail information on our website: http://easyflytracker.cibr.ac.cn
    '''


def _duration_params_validity_judgment(params):
    '''
    duration time相关参数合法性判断。这些参数只在分析那一步会用到，所以追踪的时候不要去判断。
    要求：必须<=视频时长的一半，且>0，要不然没意义。
    :param params:
    :return:
    '''
    import cv2
    vp = params['video_path']
    cap = cv2.VideoCapture(vp)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_time = frames_num / fps / 60
    cap.release()

    err_info = 'Params ERROR! the param [{}] should > 0 and <= half the length of the video!\n' \
               'Please check your params in config.yaml: [{}]'
    duration_params = ['ana_time_duration', 'sleep_time_duration', 'angle_time_duration']
    for dp in duration_params:
        v = params[dp]
        if not (0 < v <= total_time / 2):
            print(err_info.format(dp, dp))
            exit()


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

    # 判断并创建output_dir目录
    try:
        if Path(params['output_dir']).exists():
            pass
        else:
            Path(params['output_dir']).mkdir(exist_ok=True)
            print(f"Create the output_dir: {params['output_dir']}  Done!")
    except:
        print('Please set a valid [output_dir]!')
        exit()

    # 把字符串none转变为真正的None
    for key in params:
        if isinstance(params[key], str) and params[key].lower() in ['none', 'null']:
            params[key] = None

    # 这一版这个参数没有暴露给用户，为防止后续代码出问题，这里强制给它设置个默认值0.5
    params['area_th'] = 0.5

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
    # with open(r'Z:\dataset\qususu\0809_0\cfg.yaml', 'r', encoding='utf-8') as f:
    #     params = yaml.safe_load(f)
    # res = __load_group(params)
    # exit()
    # import os, cv2
    # import sys
    # import numpy as np
    # from time import time
    #
    # for i in range(2000, 2851):
    #     print(f'{i}/2850')
    #     p1 = f'Z:/temp/region/y/{i}_1y.tif'
    #     p2 = f'Z:/temp/region/y/{i}.tif'
    #     os.rename(p1, p2)
    ...
