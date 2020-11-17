#!/GPFS/zhangli_lab_permanent/zhuqingjie/env/py3_tf2/bin/python
'''
@Time    : 20/11/13 上午 10:06
@Author  : zhuqingjie 
@User    : zhu
@FileName: cli.py
@Software: PyCharm
'''
import argparse
import inspect
from pathlib import Path
from easyFlyTracker.src_code.fly_seg import FlySeg
from easyFlyTracker.src_code.analysis import Analysis


def __args_filter(kwargs, fn):
    arg_keys = inspect.getfullargspec(fn).args
    return {k: kwargs[k] for k in kwargs if k in arg_keys}


def __fly_seg(**kwargs):
    kwargs = __args_filter(kwargs, FlySeg.__init__)
    f = FlySeg(**kwargs)
    f.run()
    f.play_and_show_trackingpoints()


def __analysis(**kwargs):
    kwargs = __args_filter(kwargs, Analysis.__init__)
    a = Analysis(**kwargs)


def __show(*args, **kwargs):
    pass


def _checkargs(p):
    if p['video_path'] is None or (not Path(p['video_path']).exists()):
        print('The [video path] is not existing, please check it!')
        exit()


def _easyFlyTracker():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help='the path of the video')
    parser.add_argument('--begin_time', type=int, default=0, help='begin calculate time point, (minute)')
    parser.add_argument('--duration_time', type=int, help='how long time to calculate, (minute)')
    args = parser.parse_args()
    params = vars(args)
    _checkargs(params)
    params.update({'save_txt_name': f'{params["video_path"]}.txt'})

    # params = {
    #     'video_path': r'D:\Pycharm_Projects\qu_holmes_su_release\tests\demo.mp4',
    #     'save_txt_name': 'qudashen.txt',
    #     'begin_time': 0,
    #     'duration_time': 1,
    #     'config_it': False,
    #     'wu': 89,
    # }
    params = __args_filter(params, FlySeg.__init__)
    __fly_seg(**params)
    print()


if __name__ == '__main__':
    _easyFlyTracker()
