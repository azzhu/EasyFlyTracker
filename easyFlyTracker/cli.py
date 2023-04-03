#!/GPFS/zhangli_lab_permanent/zhuqingjie/env/py3_tf2/bin/python
'''
@Time    : 20/11/13 上午 10:06
@Author  : zhuqingjie 
@User    : zhu
@FileName: cli.py
@Software: PyCharm
'''
from loguru import logger

from easyFlyTracker.src_code.Camera_Calibration import cam_calibration
from easyFlyTracker.src_code.fly_seg import FlySeg
from easyFlyTracker.src_code.log_about import (
    log_init,
    log_base_info,
    copy_track_file,
    copy_analysis_file,
    copy_cam_calibration_file
)
from easyFlyTracker.src_code.show import one_step_run
from easyFlyTracker.src_code.utils import (
    args_filter,
    __get_params,
    __load_group,
    _duration_params_validity_judgment,
)

logger.remove(handler_id=None)  # 不在命令行输出，该命令放在logger.add前面


# Command line 1
def easyFlyTracker_(params=None):
    if params is None:
        params = __get_params()
    params.update({'save_txt_name': f'track.txt'})

    # 添加log信息
    params.update({'log': logger})
    log_init(params, '_track')
    log_base_info(params)

    # 果蝇跟踪并保存结果
    show_track_result = params['show_track_result']
    params_filter = args_filter(params, FlySeg.__init__)
    if show_track_result:
        params_filter['skip_config'] = True
        f = FlySeg(**params_filter)
        f.play_and_show_trackingpoints()
    else:
        f = FlySeg(**params_filter)
        f.run()
        if not params_filter['skip_config']:
            f.play_and_show_trackingpoints()

    # 把一些重要信息拷贝到日志目录
    copy_track_file(params)


# Command line 2
def easyFlyTracker_analysis(params=None):
    if params is None:
        params = __get_params()
    _duration_params_validity_judgment(params)
    rois = __load_group(params)
    params.update({'rois': rois})

    # 添加log信息
    params.update({'log': logger})
    log_init(params, '_analysis')
    log_base_info(params)

    # 分析结果并展示
    one_step_run(params)

    # 把一些重要信息拷贝到日志目录
    copy_analysis_file(params)


# Command line 3
def easyFlyTracker_cam_calibration(params=None):
    if params is None:
        params = __get_params()

    # 添加log信息
    params.update({'log': logger})
    log_init(params, '_cam_calibration')
    log_base_info(params)

    cam_calibration(params)

    # 把一些重要信息拷贝到日志目录
    copy_cam_calibration_file(params)


if __name__ == '__main__':
    # easyFlyTracker_()
    easyFlyTracker_analysis()
    # easyFlyTracker_cam_calibration()

    # from easyFlyTracker.src_code.log import log
    # log.name = 'zqj'
    # import easyFlyTracker.src_code.gui_config

    ...
