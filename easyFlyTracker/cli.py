#!/GPFS/zhangli_lab_permanent/zhuqingjie/env/py3_tf2/bin/python
'''
@Time    : 20/11/13 上午 10:06
@Author  : zhuqingjie 
@User    : zhu
@FileName: cli.py
@Software: PyCharm
'''
import sys
import yaml
from pathlib import Path
from easyFlyTracker.src_code.fly_seg import FlySeg
from easyFlyTracker.src_code.show import one_step_run
from easyFlyTracker.src_code.utils import (
    args_filter,
    __get_params,
    __load_group,
)


# Command line 1
def easyFlyTracker_():
    params = __get_params()
    params.update({
        'save_txt_name': f'track.txt'
    })

    # 果蝇跟踪并保存结果
    show_track_result = params['show_track_result']
    params = args_filter(params, FlySeg.__init__)
    if show_track_result:
        params['skip_config'] = True
        f = FlySeg(**params)
        f.play_and_show_trackingpoints()
    else:
        f = FlySeg(**params)
        f.run()
        f.play_and_show_trackingpoints()


# Command line 2
def easyFlyTracker_analysis():
    params = __get_params()
    rois = __load_group(params)
    params.update({
        'rois': rois,
    })

    # 分析结果并展示
    one_step_run(params)


if __name__ == '__main__':
    # easyFlyTracker_()
    easyFlyTracker_analysis()