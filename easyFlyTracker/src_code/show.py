#!/GPFS/zhangli_lab_permanent/zhuqingjie/env/py3_tf2/bin/python
'''
@Time    : 20/07/21 下午 02:03
@Author  : zhuqingjie 
@User    : zhu
@FileName: show.py
@Software: PyCharm
'''
import numpy as np
import cv2, pickle
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from easyFlyTracker.src_code.analysis import Analysis
from easyFlyTracker.src_code.utils import args_filter


class Show():
    __doc__ = \
        '''
        该类要跟analysis结合使用。
        展示Analysis计算出来的各种参数的结果；
        上一步ana分析出了结果（包含了各种参数的结果，比如不同roi的结果），
        这一步只是把这些结果给展示出来；
        
        首先会创建analysis示例，然后先分析，再根据分析出来的结果来show，        
        结果自动保存在“show_result_{视频stem}”文件夹下
        '''

    def __init__(
            self,
            video_path,
            output_dir,  # 输出文件夹
            ana_params,  # 获取analysis实例的时候需要的参数
            dish_radius_mm,
            suffix='all',  # 保存的图片结果后缀
            roi_flys_ids=None,
    ):
        '''

        :param suffix: 保存的图片或者npy文件的后缀，用于区分不同统计方式的结果
        '''
        # 初始化各种文件夹
        self.video_path = Path(video_path)
        self.output_dir = output_dir
        self.saved_dir = Path(output_dir, 'plot_images')
        self.saved_dir_npys = Path(self.saved_dir, '.npys')
        self.saved_dir.mkdir(exist_ok=True)
        self.saved_dir_npys.mkdir(exist_ok=True)

        self.video_stem = Path(video_path).stem
        cap = cv2.VideoCapture(str(self.video_path))
        self.fps = round(cap.get(cv2.CAP_PROP_FPS))  # 从opencv得到的帧率不太准，实际是30，opencv给的却是29.99，对后续计算有影响，所以取整
        cap.release()
        self.saved_suffix = suffix

        # roi_flys
        res = np.load(Path(self.output_dir, '.cache', 'track.npy'))
        if roi_flys_ids == None:
            self.roi_flys_list = np.array([True] * len(res))
        else:
            self.roi_flys_list = np.array([False] * len(res))
            self.roi_flys_list[roi_flys_ids] = True
        self.roi_flys_id = [i for i, r in enumerate(self.roi_flys_list) if r]

        # 计算比例尺
        config_pk = pickle.load(open(Path(self.output_dir, 'config.pkl'), 'rb'))
        config_pk = np.array(config_pk)
        # self.cps = config_pk[:, :2]
        self.dish_radius = int(round(float(np.mean(config_pk[:, -1]))))
        self.sacle = dish_radius_mm / self.dish_radius
        # self.sacle = 1.

        # 获取视频对应的Analysis实例
        self.ana = Analysis(**ana_params)

    def SHOW_avg_dist_per10min(self):
        '''
        十分钟一个值，
        :return:
        '''
        datas_paths = self.ana.PARAM_speed_displacement(redo=True)  # 不调缓存，重新计算
        datas = np.load(datas_paths[1])
        datas *= self.sacle
        plt.close()
        plt.rcParams['figure.figsize'] = (15.0, 8.0)
        plt.grid(linewidth=1)
        plt.xlabel('time')
        plt.ylabel('distances (mm)')
        plt.title('Average distances of flies in every duration at different time')
        plt.plot(datas, label=self.video_stem)
        plt.legend(loc='upper left')
        # plt.show()
        plt.savefig(str(Path(self.saved_dir, f'avg_dist_per_x_min_{self.saved_suffix}.png')))
        np.save(str(Path(self.saved_dir_npys, f'avg_dist_per_x_min_{self.saved_suffix}.npy')), datas)
        df = pd.DataFrame(data=datas, columns=[self.saved_suffix])
        df.to_excel(Path(self.output_dir, f'avg_dist_per_x_min_{self.saved_suffix}.xlsx'))

    def SHOW_dist_change_per_h(self):
        da = self.ana.PARAM_dist_per_h()
        base_value = da[0]
        vs = [(d - base_value) * self.sacle for d in da]
        plt.close()
        plt.rcParams['figure.figsize'] = (15.0, 8.0)
        plt.grid(linewidth=1)
        plt.xlabel('time (h)')
        plt.ylabel('distances (mm)')
        plt.title('Δ distance/hour compared with the first hour  (Distance(i)-Distance (0)) ')
        plt.plot(vs, label=self.video_stem)
        plt.legend(loc='upper left')
        # plt.show()
        plt.savefig(str(Path(self.saved_dir, f'dist_change_per_h_{self.saved_suffix}.png')))
        np.save(str(Path(self.saved_dir_npys, f'dist_change_per_h_{self.saved_suffix}.npy')), vs)
        df = pd.DataFrame(data=vs)
        df.to_excel(Path(self.output_dir, f'dist_change_per_h_{self.saved_suffix}.xlsx'))

    def SHOW_in_centre_prob_per_h(self):
        datas_path = self.ana.PARAM_region_status()
        da = np.load(datas_path)
        da = da * \
             np.tile(self.roi_flys_list[:, np.newaxis], (1, da.shape[1]))
        duration_frames = self.fps * 60 * 60
        in_centre_prob_per_h = []
        for i in range(0, da.shape[1], duration_frames):
            in_centre_prob_per_h.append(
                np.mean(da[:, i:i + duration_frames]) / duration_frames / self.ana.roi_flys_nubs)
        plt.close()
        plt.rcParams['figure.figsize'] = (15.0, 8.0)
        plt.grid(linewidth=1)
        plt.xlabel('time (h)')
        plt.ylabel('prob')
        plt.title('Ratio of Center Time to All Time per hour')
        plt.plot(in_centre_prob_per_h, label=self.video_stem)
        plt.legend(loc='upper left')
        # plt.show()
        plt.savefig(str(Path(self.saved_dir, f'in_centre_prob_per_h_{self.saved_suffix}.png')))
        np.save(str(Path(self.saved_dir_npys, f'in_centre_prob_per_h_{self.saved_suffix}.npy')), in_centre_prob_per_h)
        df = pd.DataFrame(data=in_centre_prob_per_h)
        df.to_excel(Path(self.output_dir, f'in_centre_prob_per_h_{self.saved_suffix}.xlsx'))

    def SHOW_sleep_time_per_h(self):
        '''
        每小时的睡眠总时间/有效果蝇数
        单位：分钟
        :return:
        '''
        datas_path = self.ana.PARAM_sleep_status()
        da = np.load(datas_path)
        da = da * \
             np.tile(self.roi_flys_list[:, np.newaxis], (1, da.shape[1]))
        duration_second = 60 * 60
        sleep_time_per_h = []
        for i in range(0, da.shape[1], duration_second):
            sleep_time_per_h.append(np.sum(da[:, i:i + duration_second]) / self.ana.roi_flys_nubs / 60)
        plt.close()
        plt.rcParams['figure.figsize'] = (15.0, 8.0)
        plt.grid(linewidth=1)
        plt.xlabel('time (h)')
        plt.ylabel('sleep time (minute)')
        plt.title('Sleep time of per flies per hour')
        plt.plot(sleep_time_per_h, label=self.video_stem)
        plt.legend(loc='upper left')
        # plt.show()
        plt.savefig(str(Path(self.saved_dir, f'sleep_time_per_h_{self.saved_suffix}.png')))
        np.save(str(Path(self.saved_dir_npys, f'sleep_time_per_h_{self.saved_suffix}.npy')), sleep_time_per_h)
        df = pd.DataFrame(data=sleep_time_per_h)
        df.to_excel(Path(self.output_dir, f'sleep_time_per_h_{self.saved_suffix}.xlsx'))

    def show_all(self):
        self.SHOW_avg_dist_per10min()
        # self.SHOW_dist_change_per_h()
        # self.SHOW_in_centre_prob_per_h()
        # self.SHOW_sleep_time_per_h()


def merge_result(params):
    suffixs = [v[-1] for v in params['rois']]
    prefixs = [
        'avg_dist_per_x_min',
        # 'dist_change_per_h',
        # 'in_centre_prob_per_h',
        # 'sleep_time_per_h',
    ]
    dir = Path(params['output_dir'], 'plot_images', '.npys')
    dst_dir = Path(params['output_dir'], 'plot_images')
    for pre in prefixs:
        plt.close()
        plt.rcParams['figure.figsize'] = (15.0, 8.0)
        plt.grid(linewidth=1)
        # 目前只有一个指标，所以默认这个，如果后续有多个，这个地方要改
        plt.title('Average distances of flies in every duration at different time')
        das = [np.load(Path(dir, f'{pre}_{suf}.npy')) for suf in suffixs]
        for da, lb in zip(das, suffixs):
            da = np.squeeze(da)
            plt.plot(da, label=' ' + str(lb))
        plt.legend(loc='upper left')
        plt.savefig(str(Path(dst_dir, f'{pre}_merge.png')))
        np.save(str(Path(dir, f'{pre}_merge.npy')), das)
        ...  # 顺便保存个merge的excel
        df = pd.DataFrame(np.array(das).T, columns=suffixs)
        df.to_excel(Path(params['output_dir'], f'{pre}_merge.xlsx'))


def one_step_run(params):
    '''
    Show类只针对一次roi的运算，多个区域的roi运算要分多次运行
    :param params:
    :return:
    '''
    rois = params['rois']

    for ids, flag in rois:
        ana_params = args_filter(params, Analysis)
        ana_params['roi_flys_flag'] = flag
        ana_params['roi_flys_ids'] = ids
        show_params = args_filter(params, Show)
        show_params['roi_flys_ids'] = ids
        show_params['suffix'] = flag
        show_params['ana_params'] = ana_params
        print(f'---------- {flag} ----------')
        print(ids)
        s = Show(**show_params)
        s.show_all()

    if len(rois) > 1:
        merge_result(params)


if __name__ == '__main__':
    exit()

    rois = [
        [[0, 1, 2, 3], '1'],
        [[4, 5, 6, 7], '2'],
    ]
    ana_params = {
        'video_path': r'D:\Pycharm_Projects\qu_holmes_su_release\tests\demo.mp4',
        'roi_flys_flag': '1', 'area_th': 0.5, 'ana_time_duration': 0.5,
    }

    for ids, flag in rois:
        ana_params['roi_flys_flag'] = flag
        ana_params['roi_flys_ids'] = ids
        print(f'---------- {flag} ----------')
        s = Show(video_path=r'D:\Pycharm_Projects\qu_holmes_su_release\tests\demo.mp4',
                 roi_flys_ids=ids,
                 suffix=flag,
                 ana_params=ana_params,
                 dish_radius_mm=10)
        s.show_all()
