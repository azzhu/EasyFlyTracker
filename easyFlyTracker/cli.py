#!/GPFS/zhangli_lab_permanent/zhuqingjie/env/py3_tf2/bin/python
'''
@Time    : 20/11/13 上午 10:06
@Author  : zhuqingjie 
@User    : zhu
@FileName: cli.py
@Software: PyCharm
'''
import easyFlyTracker as ft
import argparse


def _easyFlyTracker():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--total', default=100)
    args = parser.parse_args()
    print("获取命令行传参")
    t = args.total
    t = int(t)
    import time
    pbar = ft.Pbar(total=t)
    for i in range(t):
        time.sleep(0.2)
        pbar.update()
    print()


if __name__ == '__main__':
    _easyFlyTracker()

