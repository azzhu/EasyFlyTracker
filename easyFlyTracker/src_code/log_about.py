#! python
# @Time    : 23/03/29 下午 04:23
# @Author  : azzhu 
# @FileName: log_init.py
# @Software: PyCharm
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import psutil

from easyFlyTracker import __version__ as version


# 返回当前时间字符串
def get_time_now():
    t = datetime.now()
    return f'{t.year}{t.month:02d}{t.day:02d}_{t.hour:02d}{t.minute:02d}{t.second:02d}'


# 初始化log文件
def log_init(params, flag):
    logger = params['log']
    output_dir = params['output_dir']

    logdir = Path(output_dir) / '.log'
    logdir.mkdir(exist_ok=True)

    # # 删除之前的log
    # oldlogs = list(logdir.glob(f'*{flag}'))
    # for oldlog in oldlogs:
    #     shutil.rmtree(str(oldlog))

    logdir = logdir / f'{get_time_now()}{flag}'
    logdir.mkdir(exist_ok=True)
    logger.add(logdir / 'log.txt')
    params.update({'log_dir': logdir})


# track完后拷贝一些文件到log目录
def copy_track_file(params):
    outdir = Path(params['output_dir'])
    filespath = [
        outdir / 'config.bmp',
        outdir / 'config.pkl',
        outdir / '.cache' / 'background_image.bmp',
        params['config_file_path'],
    ]
    dstdir = params['log_dir']
    for f in filespath:
        if Path(f).exists():
            shutil.copy(str(f), str(dstdir))

    # 等分取10帧frame保存
    saved_dir = Path(dstdir) / 'ten_frames'
    saved_dir.mkdir(exist_ok=True)
    vp = params['video_path']
    cap = cv2.VideoCapture(vp)
    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameids = np.round(np.linspace(0, frame_num - 1, 10)).astype(int)
    for fid in frameids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        _, frame = cap.read()
        cv2.imwrite(str(saved_dir / f'{fid}.tif'), frame)
    cap.release()


# analysis完后拷贝一些文件到log目录
def copy_analysis_file(params):
    filespath = [
        params['config_file_path'],
        params['group_path'],
    ]
    dstdir = params['log_dir']
    for f in filespath:
        if Path(f).exists():
            shutil.copy(str(f), str(dstdir))


# cam_calibration完后拷贝一些文件到log目录
def copy_cam_calibration_file(params):
    filespath = [
        params['config_file_path'],
    ]
    dstdir = params['log_dir']
    for f in filespath:
        if Path(f).exists():
            shutil.copy(str(f), str(dstdir))


cai = \
    '''
    ⠀⠀⠀⠀⠰⢷⢿⠄
    ⠀⠀⠀⠀⠀⣼⣷⣄
    ⠀⠀⣤⣿⣇⣿⣿⣧⣿⡄
    ⢴⠾⠋⠀⠀⠻⣿⣷⣿⣿⡀
    ○ ⠀⢀⣿⣿⡿⢿⠈⣿
    ⠀⠀⠀⢠⣿⡿⠁⠀⡊⠀⠙
    ⠀⠀⠀⢿⣿⠀⠀⠹⣿
    ⠀⠀⠀⠀⠹⣷⡀⠀⣿⡄
    ⠀⠀⠀⠀⣀⣼⣿⠀⢈⣧.
    '''


# 记录一些基本信息，比如系统信息、软件环境信息等
def log_base_info(params):
    logger = params['log']
    logger.info(cai)
    logger.info(f'Created at {get_time_now()}')

    # 记录系统信息
    mem = psutil.virtual_memory()
    pyv = sys.version_info
    l = 30
    sys_info = f"\n=================== Base Information =================== \n" \
               f"{'OS name:':<{l}s}{platform.platform()}\n" \
               f"{'OS version:':<{l}s}{platform.version()}\n" \
               f"{'OS architecture:':<{l}s}{platform.architecture()}\n" \
               f"{'Computer type:':<{l}s}{platform.machine()}\n" \
               f"{'Computer node:':<{l}s}{platform.node()}\n" \
               f"{'Computer CPU:':<{l}s}{platform.processor()}\n" \
               f"{'Memory total:':<{l}s}{float(mem.total) / 1024 / 1024 / 1024:.4f} GB\n" \
               f"{'Memory used:':<{l}s}{float(mem.used) / 1024 / 1024 / 1024:.4f} GB\n" \
               f"{'Memory free:':<{l}s}{float(mem.free) / 1024 / 1024 / 1024:.4f} GB\n" \
               f"{'Python version:':<{l}s}{pyv[0]}.{pyv[1]}.{pyv[2]} releaselevel:{pyv[3]} serial:{pyv[4]}\n" \
               f"{'EasyFlyTracker version:':<{l}s}{version}\n" \
               f"=================== Base Information =================== "
    logger.info(sys_info)

    # 记录配置文件信息
    info = f"\n=================== Configuration Information =================== \n"
    for k, v in params.items():
        k = f'{k}:'
        info += f'{k:<30s}{v}\n'
    info += f"=================== Configuration Information =================== "
    logger.info(info)

    # 记录视频参数信息
    info = f"\n=================== Video Information =================== \n"
    vp = params['video_path']
    if not Path(vp).exists():
        logger.error('the video file not exists, exit!')
        print('the video file not exists, exit!')
        exit()
    cap = cv2.VideoCapture(str(vp))
    info += f'{"Video FRAME_COUNT:":<24s}{cap.get(cv2.CAP_PROP_FRAME_COUNT)}\n'
    info += f'{"Video FRAME_WIDTH:":<24s}{cap.get(cv2.CAP_PROP_FRAME_WIDTH)}\n'
    info += f'{"Video FRAME_HEIGHT:":<24s}{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}\n'
    info += f'{"Video FPS:":<24s}{cap.get(cv2.CAP_PROP_FPS)}\n'
    # info += f'{"Video FRAME_TYPE:":<24s}{cap.get(cv2.CAP_PROP_FRAME_TYPE)}\n'
    info += f"=================== Video Information =================== "
    logger.info(info)


if __name__ == '__main__':
    print(get_time_now())
    ...
