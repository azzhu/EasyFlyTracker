#!/GPFS/zhangli_lab_permanent/zhuqingjie/env/py3_tf2/bin/python
'''
@Time    : 20/09/27 下午 04:28
@Author  : zhuqingjie 
@User    : zhu
@FileName: load_configyaml.py
@Software: PyCharm
'''
'''
config.yaml为最原始的配置文件，但是不能灵活配置，
该脚本进一步对原始配置文件做进一步解析，
比如video_path相对路径改为绝对路径，
并且视服务器或者本机返回不同的路径。
等等
'''
import math
from pathlib import Path
import platform
import numpy as np
import yaml


def load_config(set_video_path=None, set_roi=None):
    '''
    加上set_video_path这个参数可以动态配置视频路径，方便数据批处理操作
    :param set_roi: str的list，视频中三列的标记
    :param set_video_path:
    :return:
    '''
    cfg = yaml.safe_load(open('config.yaml', 'rb'))
    if set_video_path != None:
        cfg['guoyingfang_video_path'] = set_video_path
    if set_roi != None:
        for i, r in enumerate(set_roi):
            cfg['rois'][i + 1][-1] = r
    cfg['guoyingfang_video_path'] = cfg['guoyingfang_video_path'].replace(' ', '')

    video_name = str(Path(cfg['guoyingfang_video_path']).name)
    video_dir = video_name[4:8]
    if 'Windows' in platform.platform():
        video_path = f'Z:/dataset/qususu/{video_dir}/{video_name}'
        cfg['mapxy_path'] = f'Z:/dataset/qususu/{cfg["mapxy_path"]}'
    else:
        video_path = f'/home/zhangli_lab/zhuqingjie/dataset/qususu/{video_dir}/{video_name}'
        cfg['mapxy_path'] = f'/home/zhangli_lab/zhuqingjie/dataset/qususu/{cfg["mapxy_path"]}'
    cfg['video_path'] = video_path
    cfg['video_name'] = video_name
    cfg['video_dir'] = video_dir

    cfg['roi_flys_mask_arry'] = np.ones([cfg['h_num'], cfg['w_num']], np.bool)
    # cfg['roi_flys_flag'] = [v[-1] for v in cfg['rois']]

    return cfg


if __name__ == '__main__':
    cfg = load_config(set_video_path='20201021_w1118_MPHcurve1_018/202010211040.avi', set_roi=['C', 'nn', 'io'])
    print()
