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
from easyFlyTracker.src_code.utils import args_filter
from easyFlyTracker.src_code.fly_seg import FlySeg
from easyFlyTracker.src_code.analysis import Analysis
from easyFlyTracker.src_code.show import Show


def __fly_seg(**kwargs):  # 果蝇跟踪并保存结果
    show_track_result = kwargs['show_track_result']
    kwargs = args_filter(kwargs, FlySeg.__init__)
    if show_track_result:
        f = FlySeg(**kwargs, config_it=False)
        f.play_and_show_trackingpoints()
    else:
        f = FlySeg(**kwargs)
        f.run()


def __analysis(**kwargs):  # 分析结果并展示
    kwargs = args_filter(kwargs, Analysis.__init__)
    a = Analysis(**kwargs)


def easyFlyTracker_():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help='the path of the video',
                        # default=r'D:\Pycharm_Projects\qu_holmes_su_release\tests\demo.mp4',
                        default=None,
                        )
    parser.add_argument('--begin_time', type=int, default=0,
                        help='optional, default 0. begin calculate time point, (minute)')
    parser.add_argument('--duration_time', type=int, default=None,
                        help='optional, how long time to calculate, (minute)')
    parser.add_argument('--show_track_result', type=bool, default=True,
                        help='optional, default False. show track result')
    args = parser.parse_args()
    params = vars(args)

    if params['video_path'] is None or (not Path(params['video_path']).exists()):
        print('The [video path] is not existing, please check it!')
        exit()

    params.update({'save_txt_name': f'{Path(params["video_path"]).stem}.txt'})
    __fly_seg(**params)


def easyFlyTracker_analysis():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help='the path of the video',
                        default=r'D:\Pycharm_Projects\qu_holmes_su_release\tests\demo.mp4',
                        # default=None,
                        )
    parser.add_argument('--begin_time', type=int, default=0,
                        help='optional, default 0. begin calculate time point, (minute)')
    parser.add_argument('--duration_time', type=int, default=None,
                        help='optional, how long time to calculate, (minute)')
    parser.add_argument('--show_track_result', type=bool, default=True,
                        help='optional, default False. show track result')
    args = parser.parse_args()
    params = vars(args)

    if params['video_path'] is None or (not Path(params['video_path']).exists()):
        print('The [video path] is not existing, please check it!')
        exit()

    params.update({'save_txt_name': f'{Path(params["video_path"]).stem}.txt'})


if __name__ == '__main__':
    easyFlyTracker_analysis()
