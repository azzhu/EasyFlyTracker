#!/GPFS/zhangli_lab_permanent/zhuqingjie/env/py3_tf2/bin/python
'''
@Time    : 20/07/21 下午 03:25
@Author  : zhuqingjie 
@User    : zhu
@FileName: analysis.py
@Software: PyCharm
'''
import numpy as np
import cv2, time, random, math
from pathlib import Path
# import scipy.signal
import pandas as pd
import pickle
# from easyFlyTracker.src_code.fly_seg import FlySeg
from easyFlyTracker.src_code.utils import Pbar
from easyFlyTracker.src_code.utils import NumpyArrayHasNanValuesExceptin


class Analysis():
    '''
    分析实验结果，
    这里计算出来的结果涉及到长度的单位都是像素，比例尺在后续分析中使用，在这统一不使用比例尺。
    '''
    __doc__ = 'ana'

    def __init__(
            self,
            video_path,  # 视频路径
            output_dir,  # 输出文件夹
            roi_flys_flag,
            area_th,  # 内圈面积占比
            roi_flys_ids=None,
            ana_time_duration=10.,
            sleep_dist_th_per_second=5,
            sleep_time_th=300,  # 每秒睡眠状态持续多久算是真正的睡眠
    ):
        # load cps and radius
        config_pk = pickle.load(open(Path(Path(video_path).parent, 'config.pkl'), 'rb'))
        config_pk = np.array(config_pk)
        self.cps = config_pk[:, :2]
        self.dish_radius = int(round(float(np.mean(config_pk[:, -1]))))

        self.video_path = Path(video_path)
        self.roi_flys_flag = roi_flys_flag
        self.ana_time_duration = ana_time_duration
        self.sleep_dist_th_per_second = sleep_dist_th_per_second
        self.sleep_time_th = sleep_time_th
        self.region_radius = int(round(math.sqrt(area_th) * self.dish_radius))

        # 初始化各种文件夹
        self.res_dir = Path(output_dir)  # 保存用户需要的结果
        self.cache_dir = Path(self.res_dir, '.cache')  # 保存程序计算的中间结果
        self.saved_dir = Path(self.cache_dir, 'analysis_result')  # analysis计算出的结果
        self.npy_file_path = Path(self.cache_dir, f'track.npy')
        self.npy_file_path_cor = Path(self.cache_dir, f'track_cor.npy')
        self.speeds_npy = Path(self.saved_dir, 'all_fly_speeds_per_frame.npy')
        self.dist_npy = Path(self.saved_dir, 'all_fly_dist_per_frame.npy')
        self.cache_dir.mkdir(exist_ok=True)
        self.saved_dir.mkdir(exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        self.fps = round(cap.get(cv2.CAP_PROP_FPS))
        self.video_frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # 果蝇总数，这个结果是综合了roi_flys_mask_arry和Dish_exclude后的结果
        if roi_flys_ids == None:
            self.roi_flys_list = np.array([True] * len(self.cps))
        else:
            self.roi_flys_list = np.array([False] * len(self.cps))
            self.roi_flys_list[roi_flys_ids] = True
        self.roi_flys_id = [i for i, r in enumerate(self.roi_flys_list) if r]
        self.roi_flys_nubs = self.roi_flys_list.sum()

        # 初始化加载某些数据
        self._get_all_res()
        self._get_speed_perframe_dist_perframe()

    def _get_all_res(self):
        if self.npy_file_path_cor.exists():
            self.all_datas = np.load(self.npy_file_path_cor)
        else:
            res = np.load(self.npy_file_path)
            self.all_datas = np.transpose(res, [1, 0, 2])
            self._cor()
            np.save(self.npy_file_path_cor, self.all_datas)

    def _cor(self):
        def _correction(l):
            '''
            对一个向量进行校正，以下规则：
            1，不含有-1，不作处理直接return；
            2，含有-1，使用线性插值去掉-1。
            :param l:
            :return:
            '''
            # l: 一个int或者float类型的数值list，形如[-1, -1, 88, 90, -1, -1, -1, 100]，其中-1为异常点
            if not (np.array(l) == -1).any():  # 不含有-1直接return
                return l

            # 因为pandas的方法不能对前面的坑进行插值，所以先把前面的坑补全了
            if l[0] < 0:
                for i in range(len(l)):
                    if l[i] > 0:
                        l[:i] = l[i]
                        break
            l = np.where(l < 0, np.nan, l)
            df = pd.DataFrame(data=l)
            df.interpolate(method="linear", inplace=True)
            return df.values[:, 0]

        def correction2D(ps):
            return list(zip(_correction(ps[:, 0]), _correction(ps[:, 1])))

        res = []
        for ps in self.all_datas:
            # 判断是不是空盘（空盘值全部为(-1,-1)），空盘直接返回
            if ps.min() == -1 and ps.max() == -1:
                res.append(ps)
            else:
                res.append(correction2D(ps))
        res = np.array(res)
        if np.isnan(res).sum() != 0:
            raise NumpyArrayHasNanValuesExceptin(res)
        self.all_datas = res

    def _get_speed_perframe_dist_perframe(self, redo=False):
        if self.speeds_npy.exists() and self.dist_npy.exists() and not redo:
            self.all_fly_speeds_per_frame = np.load(self.speeds_npy)
            self.all_fly_dist_per_frame = np.load(self.dist_npy)
            return
        fn = lambda x, y: math.sqrt(pow(x, 2) + pow(y, 2))
        fn2 = lambda ps, k: fn(ps[k][0] - ps[k + 1][0],  # 两点之间的距离
                               ps[k][1] - ps[k + 1][1])
        all_fly_speeds = []  # 长度等于帧数
        all_fly_displacement = []  # 长度等于帧数减一
        mperframe = 1 / self.fps
        for fly in self.all_datas:
            # if not exc:
            #     all_fly_displacement.append([0] * (self.all_datas.shape[1] - 1))
            #     all_fly_speeds.append([0] * self.all_datas.shape[1])
            #     continue
            ds = [fn2(fly, i) for i in range(len(fly) - 1)]
            all_fly_displacement.append(ds)
            ds = [ds[0]] + ds + [ds[-1]]
            speed = [(ds[i] + ds[i + 1]) / (2 * mperframe) for i in range(len(ds) - 1)]
            all_fly_speeds.append(speed)
        if np.isnan(all_fly_speeds).sum() != 0:
            raise NumpyArrayHasNanValuesExceptin(all_fly_speeds)
        if np.isnan(all_fly_displacement).sum() != 0:
            raise NumpyArrayHasNanValuesExceptin(all_fly_displacement)
        np.save(self.speeds_npy, all_fly_speeds)
        np.save(self.dist_npy, all_fly_displacement)
        self.all_fly_speeds_per_frame = np.array(all_fly_speeds)
        self.all_fly_dist_per_frame = np.array(all_fly_displacement)

    def PARAM_speed_displacement(self, redo=False):
        '''
        计算两个参数：
        time_duration_stat_speed：10分钟总速度/帧数/果蝇个数；
        time_duration_stat_displacement：10分钟总位移/果蝇个数
        :return: 返回npy路径
        '''
        speed_npy = Path(self.saved_dir, f'time_duration_stat_speed_{self.roi_flys_flag}.npy')
        disp_npy = Path(self.saved_dir, f'time_duration_stat_displacement_{self.roi_flys_flag}.npy')
        if speed_npy.exists() and disp_npy.exists() and not redo:
            return speed_npy, disp_npy
        duration_frames = int(round(self.ana_time_duration * 60 * self.fps))
        frame_start_ind = list(range(0, self.all_datas.shape[1], duration_frames))
        all_fly_speeds = self.all_fly_speeds_per_frame * \
                         np.tile(self.roi_flys_list[:, np.newaxis],
                                 (1, self.all_fly_speeds_per_frame.shape[1]))
        all_fly_displacement = self.all_fly_dist_per_frame * \
                               np.tile(self.roi_flys_list[:, np.newaxis],
                                       (1, self.all_fly_dist_per_frame.shape[1]))

        time_duration_stat_speed = []  # 10分钟总速度/帧数/果蝇个数
        time_duration_stat_displacement = []  # 10分钟总位移/果蝇个数
        for ind in frame_start_ind:
            time_duration_stat_speed.append(
                all_fly_speeds[:, ind:ind + duration_frames].sum() / duration_frames / self.roi_flys_nubs)
            time_duration_stat_displacement.append(
                all_fly_displacement[:, ind:ind + duration_frames].sum() / self.roi_flys_nubs)
        np.save(speed_npy, time_duration_stat_speed)
        np.save(disp_npy, time_duration_stat_displacement)
        return speed_npy, disp_npy

    def PARAM_dist_per_h(self):
        '''
        每小时的移动总距离/果蝇数量
        :return: list
        '''
        self._get_speed_perframe_dist_perframe()
        fps = self.fps
        dist_per_h = []
        da = self.all_fly_dist_per_frame * \
             np.tile(self.roi_flys_list[:, np.newaxis],
                     (1, self.all_fly_dist_per_frame.shape[1]))
        duration_frames = int(round(fps * 60 * 60))
        for i in range(0, da.shape[1], duration_frames):
            dist_per_h.append(np.sum(da[:, i:i + duration_frames]) / self.roi_flys_nubs)
        return dist_per_h

    def PARAM_sleep_status(self, redo=False):
        '''
        计算每一秒的睡眠状态，返回保存的npy路径
        :param redo:
        :return:
        '''
        npy_path = Path(self.saved_dir, f'sleep_status_per_second.npy')
        if npy_path.exists() and not redo:
            return str(npy_path)
        self._get_speed_perframe_dist_perframe()
        fps = self.fps
        all_dist_per_s = []
        for i in range(self.all_fly_dist_per_frame.shape[0]):
            dist_per_s = []
            for j in range(0, self.all_fly_dist_per_frame.shape[1], fps):
                dist_per_s.append(np.sum(self.all_fly_dist_per_frame[i, j:j + fps]))
            all_dist_per_s.append(dist_per_s)

        # sp = self.all_fly_dist_per_frame.shape
        # all_fly_dist_per_frame_cut = self.all_fly_dist_per_frame[:, :sp[1] - divmod(sp[1], int(fps))[1]]
        # all_dist_per_s = all_fly_dist_per_frame_cut.reshape([sp[0], -1, int(fps)])
        # all_dist_per_s = np.sum(all_dist_per_s, axis=-1)

        sleep_dist_th = self.sleep_dist_th_per_second
        all_sleep_status_per_s = np.array(all_dist_per_s) < sleep_dist_th
        self.all_sleep_status_per_s = all_sleep_status_per_s
        # all_sleep_status_per_s = np.delete(all_sleep_status_per_s, exclude_ids, axis=0)
        sleep_time_th = self.sleep_time_th
        all_sleep_status = []
        for k, sleep_status_per_s in enumerate(all_sleep_status_per_s):
            sleep_time = 0  # 用于保存截止当前秒已经睡了多久（单位秒）
            sleep_status_per_s = np.concatenate(
                [sleep_status_per_s, np.array([False])])  # 在末尾加一个false，防止末尾是True时遍历结束时无法判断睡眠
            sleep_status = np.zeros([len(sleep_status_per_s) - 1, ], np.bool)  # 新创建的list，用于保存睡眠状态
            for i, ss in enumerate(sleep_status_per_s):
                if ss:
                    sleep_time += 1
                else:
                    # 到没睡的时候都判断一下，上一刻截止是不是满足睡眠条件
                    if sleep_time >= sleep_time_th:
                        sleep_status[i - sleep_time:i] = True
                    sleep_time = 0
            all_sleep_status.append(sleep_status)
        np.save(str(npy_path), np.array(all_sleep_status))
        return str(npy_path)

    def PARAM_region_status(self):
        '''
        统计每一帧是否在内圈的结果，在为True，不在为False。注意，被排除的果盘也被置为False了
        :return: 保存的npy路径  （果蝇数,帧数）
        '''
        region_status_npy = Path(self.saved_dir, f'region_status.npy')
        if Path(region_status_npy).exists():
            self.all_region_status = np.load(region_status_npy)
            return str(region_status_npy)

        cps = self.cps
        all_datas = self.all_datas.astype(np.float16)
        all_region_status = []
        print('get_region_status:')
        pbar = Pbar(total=len(cps))
        for i, (cp, da) in enumerate(zip(cps, all_datas)):
            dist_to_cp = lambda x: math.sqrt(math.pow(x[0] - cp[0], 2) + math.pow(x[1] - cp[1], 2))
            region_status = np.array([dist_to_cp(p) < self.region_radius for p in da])
            all_region_status.append(region_status)
            pbar.update()
        pbar.close()
        self.all_region_status = np.array(all_region_status)
        np.save(region_status_npy, self.all_region_status)
        return str(region_status_npy)


if __name__ == '__main__':
    a = Analysis(
        video_path=r'D:\Pycharm_Projects\qu_holmes_su_release\tests\demo.mp4',
        roi_flys_flag='1', area_th=0.5, ana_time_duration=0.5,
    )
    a.PARAM_speed_displacement()
