#!/GPFS/zhangli_lab_permanent/zhuqingjie/env/py3_tf2/bin/python
'''
@Time    : 20/11/13 上午 10:06
@Author  : zhuqingjie 
@User    : zhu
@FileName: cli.py
@Software: PyCharm
'''
import argparse
from pathlib import Path
from easyFlyTracker.src_code.fly_seg import FlySeg


def __fly_seg(*args, **kwargs):
    f = FlySeg(*args, **kwargs)
    f.run()
    f.play_and_show_trackingpoints()


def __analysis(*args, **kwargs):
    pass


def __show(*args, **kwargs):
    pass


def _easyFlyTracker():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default=None,
                        help='the path of the video')
    parser.add_argument('--save_txt_name', type=str, default='0000.txt',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--begin_time', type=int, default=0,
                        help='disables CUDA training')
    parser.add_argument('--duration_time', type=int, default=1,
                        help='For Saving the current Model')
    args = parser.parse_args()
    if args.video_path is None or (not Path(args.video_path).exists()):
        print('The video path is not existing, please check it!')
    exit()

    pas = vars(args)
    pas.update({'save_txt_name': f'{pas["video_path"]}.txt'})

    params = {
        'video_path': r'D:\Pycharm_Projects\qu_holmes_su_release\tests\demo.mp4',
        'save_txt_name': 'qudashen.txt',
        'begin_time': 0,
        'duration_time': 1,
        'config_it': False,
    }
    __fly_seg(**params)
    print()


if __name__ == '__main__':
    _easyFlyTracker()
