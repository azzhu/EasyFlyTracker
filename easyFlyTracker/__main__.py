#! python
# @Time    : 23/03/08 下午 05:23
# @Author  : azzhu 
# @FileName: __main__.py
# @Software: PyCharm
import sys
from easyFlyTracker.src_code.utils import __get_params
from easyFlyTracker.cli import (
    easyFlyTracker_,
    easyFlyTracker_analysis,
    easyFlyTracker_cam_calibration,
)
import easyFlyTracker

version = easyFlyTracker.__version__

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

'''
该文件的作用是满足HELP这种用法.
'''
HELP = \
    f'''\n{cai}
    
Version: 
    {version}

Usage example:
    python -m easyFlyTracker [config_path]      # Corresponding terminal command: easyFlyTracker [config_path]
    python -m easyFlyTracker -a [config_path]   # Corresponding terminal command: easyFlyTracker_analysis [config_path]
    python -m easyFlyTracker -c [config_path]   # Corresponding terminal command: easyFlyTracker_cam_calibration [config_path]

You can find more detail information on our website: http://easyflytracker.cibr.ac.cn
\n'''

print(HELP)
exit()

args = sys.argv

if len(args) == 1:  # 没有附加参数
    print(HELP)

elif len(args) == 2:  # 附加一个参数，对应easyFlyTracker
    cfg_p = args[-1]
    params = __get_params(cfg_p)
    easyFlyTracker_(params)

elif len(args) == 3:  # 附加两个参数，对应easyFlyTracker_analysis或者easyFlyTracker_cam_calibration
    mode = args[1]
    cfg_p = args[-1]
    if mode == '-a':
        params = __get_params(cfg_p)
        easyFlyTracker_analysis(params)
    elif mode == '-c':
        params = __get_params(cfg_p)
        easyFlyTracker_cam_calibration(params)
    else:
        print('params error: the first param should "-a" or "-c" when you input two params.')
        print(HELP)

else:
    print('params error!')
    print(HELP)
