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
import scipy.signal
import pandas as pd
from easyFlyTracker.src_code.fly_seg import FlySeg
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
            h_num, w_num,  # 盘子是几乘几的
            roi_flys_flag,
            roi_flys_mask_arry,
            dish_radius,  # 每个孔的半径，像素
            area_th,  # 内圈面积占比
            duration_time=2,  # 要持续多长时间
            ana_time_duration=10,
            minR_maxR_minD=(40, 50, 90),
            sleep_dist_th_per_second=5,
            sleep_time_th=300,  # 每秒睡眠状态持续多久算是真正的睡眠
            dish_exclude=None,  # 排除的特殊圆盘，比如空盘、死果蝇等情况,可以一维或者（h_num, w_num）
    ):
        self.video_path = video_path
        if type(dish_exclude) == np.ndarray:
            self.dish_exclude = dish_exclude.reshape([-1])
        elif type(dish_exclude) == list:
            if type(dish_exclude[0]) == list:
                self.dish_exclude = np.ones([h_num, w_num], np.bool)
                for de in dish_exclude: self.dish_exclude[de[0], de[1]] = False
                self.dish_exclude = self.dish_exclude.reshape([-1])
            else:
                self.dish_exclude = np.array([True] * h_num * w_num)
                self.dish_exclude[dish_exclude] = False
        else:
            self.dish_exclude = np.array([True] * h_num * w_num)
        self.h_num = h_num
        self.w_num = w_num
        self.duration_time = duration_time
        self.dish_radius = dish_radius
        self.roi_flys_flag = roi_flys_flag
        self.ana_time_duration = ana_time_duration
        self.minR_maxR_minD = minR_maxR_minD
        self.sleep_dist_th_per_second = sleep_dist_th_per_second
        self.sleep_time_th = sleep_time_th
        self.region_radius = int(round(math.sqrt(area_th) * self.dish_radius))

        self.res_dir = Path(Path(video_path).parent, Path(video_path).stem)
        cap = cv2.VideoCapture(video_path)
        self.fps = round(cap.get(cv2.CAP_PROP_FPS))
        self.video_frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # 果蝇总数，这个结果是综合了roi_flys_mask_arry和Dish_exclude后的结果
        self.roi_flys_list = roi_flys_mask_arry.reshape([-1, ]) * self.dish_exclude
        self.roi_flys_id = [i for i, r in enumerate(self.roi_flys_list) if r]
        self.roi_flys_nubs = self.roi_flys_list.sum()

        # 初始化保存的文件夹
        saved_dir = Path(self.res_dir, 'analysis_result')
        saved_dir.mkdir(exist_ok=True)
        self.saved_dir = saved_dir

        # 初始化加载某些数据
        self._get_all_res()
        self._get_speed_perframe_dist_perframe()

    def _get_all_res(self):
        temp_npy = f'{self.res_dir}/total.npy'
        if Path(temp_npy).exists():
            self.all_datas = np.load(temp_npy)
            return
        txts_path = [Path(self.res_dir, f'{t:0>4d}.txt') for t in range(0, 1000, self.duration_time)]
        npys_path = [Path(self.res_dir, f'{t:0>4d}.npy') for t in range(0, 1000, self.duration_time)]
        txts_path = [p for p in txts_path if p.exists()]
        npys_path = [p for p in npys_path if p.exists()]
        begin_points = [int(open(txt, 'r').readlines()[0].strip()) for txt in txts_path]
        npys = [np.load(p) for p in npys_path]

        res = np.zeros([self.video_frames_num, npys[0].shape[1], npys[0].shape[2]], npys[0].dtype)
        for npy, bp in zip(npys, begin_points):
            res[bp:bp + len(npy)] = npy
        self.all_datas = np.transpose(res, [1, 0, 2])
        self._smooth()
        np.save(temp_npy, self.all_datas)

    def _smooth(self):
        def sgolay2d_points(ps, window_size=27, order=5):
            vs0 = [p[0] for p in ps]
            vs1 = [p[1] for p in ps]
            vs0 = scipy.signal.savgol_filter(vs0, window_size, order)
            vs1 = scipy.signal.savgol_filter(vs1, window_size, order)
            vs = np.array(list(zip(vs0, vs1)))
            vs = np.round(vs, 2)
            return vs

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
        for ps, exc in zip(self.all_datas, self.dish_exclude):
            if exc:
                # 判断是不是空盘（空盘值全部为(-1,-1)），空盘直接返回
                if ps.min() == -1 and ps.max() == -1:
                    res.append(ps)
                else:
                    # res.append(sgolay2d_points(correction2D(ps))) # 有平滑
                    res.append(correction2D(ps))  # 无平滑
            else:
                res.append(ps)
        res = np.array(res)
        if np.isnan(res).sum() != 0:
            raise NumpyArrayHasNanValuesExceptin(res)
        self.all_datas = res

    def _get_speed_perframe_dist_perframe(self, redo=False):
        speeds_npy = f'{self.res_dir}/all_fly_speeds_per_frame.npy'
        dist_npy = f'{self.res_dir}/all_fly_dist_per_frame.npy'
        if Path(speeds_npy).exists() and Path(dist_npy).exists() and not redo:
            self.all_fly_speeds_per_frame = np.load(speeds_npy)
            self.all_fly_dist_per_frame = np.load(dist_npy)
            return
        fn = lambda x, y: math.sqrt(pow(x, 2) + pow(y, 2))
        fn2 = lambda ps, k: fn(ps[k][0] - ps[k + 1][0],  # 两点之间的距离
                               ps[k][1] - ps[k + 1][1])
        all_fly_speeds = []  # 长度等于帧数
        all_fly_displacement = []  # 长度等于帧数减一
        mperframe = 1 / self.fps
        for fly, exc in zip(self.all_datas, self.dish_exclude):
            if not exc:
                all_fly_displacement.append([0] * (self.all_datas.shape[1] - 1))
                all_fly_speeds.append([0] * self.all_datas.shape[1])
                continue
            ds = [fn2(fly, i) for i in range(len(fly) - 1)]
            all_fly_displacement.append(ds)
            ds = [ds[0]] + ds + [ds[-1]]
            speed = [(ds[i] + ds[i + 1]) / (2 * mperframe) for i in range(len(ds) - 1)]
            all_fly_speeds.append(speed)
        if np.isnan(all_fly_speeds).sum() != 0:
            raise NumpyArrayHasNanValuesExceptin(all_fly_speeds)
        if np.isnan(all_fly_displacement).sum() != 0:
            raise NumpyArrayHasNanValuesExceptin(all_fly_displacement)
        np.save(speeds_npy, all_fly_speeds)
        np.save(dist_npy, all_fly_displacement)
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
        # if speed_npy.exists() and disp_npy.exists() and not redo:
        #     return speed_npy, disp_npy
        self._get_speed_perframe_dist_perframe()
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
        exclude_ids = list(np.squeeze(np.argwhere(np.array(self.dish_exclude) == False)))
        # all_sleep_status_per_s = np.delete(all_sleep_status_per_s, exclude_ids, axis=0)
        sleep_time_th = self.sleep_time_th
        all_sleep_status = []
        for k, sleep_status_per_s in enumerate(all_sleep_status_per_s):
            if k in exclude_ids:
                all_sleep_status.append(np.zeros([len(sleep_status_per_s), ], np.bool))
            else:
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
        cps, _ = FlySeg._get_all_centre_points(
            self.video_path, self.h_num, self.w_num, self.minR_maxR_minD, None)
        all_datas = self.all_datas.astype(np.float16)
        all_region_status = []
        print('get_region_status:')
        pbar = Pbar(total=len(cps))
        for i, (cp, da, exc) in enumerate(zip(cps, all_datas, self.dish_exclude)):
            if exc:
                dist_to_cp = lambda x: math.sqrt(math.pow(x[0] - cp[0], 2) + math.pow(x[1] - cp[1], 2))
                region_status = np.array([dist_to_cp(p) < self.region_radius for p in da])
                all_region_status.append(region_status)
            else:
                all_region_status.append([False] * len(da))
            pbar.update()
        pbar.close()
        self.all_region_status = np.array(all_region_status)
        np.save(region_status_npy, self.all_region_status)
        return str(region_status_npy)


if __name__ == '__main__':
    import inspect

    res = inspect.getfullargspec(Analysis.__init__)
    print()
