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
from easyFlyTracker.src_code.utils import args_filter
from easyFlyTracker.src_code.fly_seg import FlySeg
from easyFlyTracker.src_code.show import one_step_run


def __get_params():
    args = sys.argv
    if len(args) == 1:
        print('please set the path of config file!')
        exit()
    cfg_p = Path(args[1])
    if not cfg_p.exists():
        print('please check that if the config file path is correct')
        exit()
    params = yaml.safe_load(open(cfg_p, 'r', encoding='utf-8'))
    vp = params['video_path']
    if vp is None or not Path(vp).exists():
        print('The [video_path] is not existing, please check it!')
        exit()
    return params


def __load_config_roi(params):
    p = params['config_roi']
    if p:  # 配置了该路径
        if Path(p).exists():
            dst_p = p
        else:
            print('the path, [config_roi], is not exists!')
            exit()
    else:  # 未配置该路径
        vp = Path(params['video_path'])
        cp = Path(vp.parent, 'config_roi.txt')
        dst_p = cp if cp.exists() else None  # 未配置也没找到，设为None

    # load config roi
    config_rois = []
    if dst_p:
        with open(dst_p, 'r') as f:
            lines = f.readlines()
            for line in lines:
                l = line.strip()
                flag, roi = l.split()
                roi = roi.split(',')
                roi = list(map(int, roi))
                config_rois.append([roi, flag])
    else:
        config_rois.append([None, 'all'])  # roi设为空的时候就是全部，色即是空，空即是色。
    return config_rois


# Command line 1
def easyFlyTracker_():
    params = __get_params()
    params.update({
        'save_txt_name': f'{Path(params["video_path"]).stem}.txt'
    })

    # 果蝇跟踪并保存结果
    show_track_result = params['show_track_result']
    params = args_filter(params, FlySeg.__init__)
    if show_track_result:
        f = FlySeg(**params, config_it=False)
        f.play_and_show_trackingpoints()
    else:
        f = FlySeg(**params)
        f.run()


# Command line 2
def easyFlyTracker_analysis():
    params = __get_params()
    rois = __load_config_roi(params)
    params.update({
        'rois': rois,
    })

    # 分析结果并展示
    one_step_run(params)


if __name__ == '__main__':
    # easyFlyTracker_()
    # easyFlyTracker_analysis()
    import tkinter