#!/GPFS/zhangli_lab_permanent/zhuqingjie/env/py3_tf2/bin/python
'''
@Time    : 20/07/21 下午 03:25
@Author  : zhuqingjie 
@User    : zhu
@FileName: analysis.py
@Software: PyCharm
'''
import pickle
import warnings
from pathlib import Path

import cv2
import math
import numpy as np
import pandas as pd

from easyFlyTracker.src_code.Camera_Calibration import Undistortion
from easyFlyTracker.src_code.utils import NumpyArrayHasNanValuesExceptin
from easyFlyTracker.src_code.utils import Pbar, equalizeHist_use_mask

warnings.filterwarnings("ignore")


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
            area_th=0.5,  # 内圈面积占比
            roi_flys_ids=None,
            ana_time_duration=10.,  # 分析移动距离的时候每个值需要统计的时间跨度
            sleep_time_duration=10.,  # 统计睡眠信息的时候每个值需要统计的时间跨度
            angle_time_duration=10,  # 统计角度变化信息的时候每个值需要统计的时间跨度
            sleep_dist_th_per_second=5,
            sleep_time_th=300,  # 每秒睡眠状态持续多久算是真正的睡眠
            Undistortion_model_path=None,  # 畸变矫正参数路径
    ):
        # 初始化各种文件夹及路径
        self.video_path = Path(video_path)
        self.res_dir = Path(output_dir)  # 保存用户需要的结果
        self.cache_dir = Path(self.res_dir, '.cache')  # 保存程序计算的中间结果
        self.saved_dir = Path(self.cache_dir, 'analysis_result')  # analysis计算出的结果
        self.npy_file_path = Path(self.cache_dir, f'track.npy')
        self.npy_file_path_cor = Path(self.cache_dir, f'track_cor.npy')
        self.move_direction_pre_frame_path = Path(self.saved_dir, 'move_direction_pre_frame.npy')
        self.fly_angles_cor_path = Path(self.saved_dir, 'fly_angles_cor.npy')
        self.speeds_npy = Path(self.saved_dir, 'all_fly_speeds_per_frame.npy')
        self.dist_npy = Path(self.saved_dir, 'all_fly_dist_per_frame.npy')
        # self.angle_changes_path = Path(self.saved_dir, 'angle_changes.npy')
        config_pkl_path = Path(self.res_dir, 'config.pkl')
        self.cache_dir.mkdir(exist_ok=True)
        self.saved_dir.mkdir(exist_ok=True)
        self.Undistortion_model_path = Undistortion_model_path

        # load cps and radius
        config_pk = np.array(pickle.load(open(config_pkl_path, 'rb'))[0])
        self.cps = config_pk[:, :2]
        self.dish_radius = int(round(float(np.mean(config_pk[:, -1]))))
        self.mask_imgs = np.load(Path(self.cache_dir, 'mask_imgs.npy'))
        self.mask_imgs = self.mask_imgs.astype(np.bool)

        self.roi_flys_flag = roi_flys_flag
        self.ana_time_duration = ana_time_duration
        self.sleep_time_duration = sleep_time_duration
        self.angle_time_duration = angle_time_duration
        self.sleep_dist_th_per_second = sleep_dist_th_per_second
        self.sleep_time_th = sleep_time_th
        self.region_radius = int(round(math.sqrt(area_th) * self.dish_radius))

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

        # load heatmap
        heatmap_path = Path(self.cache_dir, 'heatmap.npy')
        self.heatmap = np.load(heatmap_path)

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
        speed_npy = Path(self.saved_dir, f'speed_per_duration_{self.roi_flys_flag}.npy')
        disp_npy = Path(self.saved_dir, f'displacement_per_duration_{self.roi_flys_flag}.npy')
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

        # res = []
        # for ind in frame_start_ind:
        #     x = np.sum(self.all_fly_dist_per_frame[:, ind:ind + duration_frames], axis=1)
        #     res.append(x)
        # res = np.stack(res, axis=-1)
        # res = res * 0.26876426270157516
        # np.save(r'Z:\dataset\qususu\ceshishipin\v080\output_72hole_0330_v080\plot_images\qudashen.npy', res)
        # df = pd.DataFrame(data=res)
        # df.to_excel(r'Z:\dataset\qususu\ceshishipin\v080\output_72hole_0330_v080\plot_images\qudashen.xlsx')
        # exit()

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
        首先计算每一秒的睡眠状态，然后计算统计时间段（sleep_time_duration）内果蝇的总睡眠时长，计算方式为：
        所有果蝇该时间段的总睡眠时长/果蝇数量
        返回保存的npy路径
        :param redo:
        :return:
        '''
        npy_path = Path(self.saved_dir, f'sleep_time_per_duration_{self.roi_flys_flag}.npy')
        if npy_path.exists() and not redo:
            return str(npy_path)
        cache_all_sleep_status_path = Path(self.cache_dir, 'all_sleep_status.npy')

        def get_all_sleep_status(self):
            if cache_all_sleep_status_path.exists():
                return np.load(cache_all_sleep_status_path)
            self._get_speed_perframe_dist_perframe()
            fps = self.fps
            all_dist_per_s = []
            for i in range(self.all_fly_dist_per_frame.shape[0]):
                dist_per_s = []
                for j in range(0, self.all_fly_dist_per_frame.shape[1], fps):
                    dist_per_s.append(np.sum(self.all_fly_dist_per_frame[i, j:j + fps]))
                all_dist_per_s.append(dist_per_s)

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
            # 每个果蝇每秒钟的睡眠状态
            all_sleep_status = np.array(all_sleep_status)
            np.save(cache_all_sleep_status_path, all_sleep_status)
            return all_sleep_status

        all_sleep_status = get_all_sleep_status(self)
        all_sleep_status = all_sleep_status[self.roi_flys_id]

        dt = int(round(self.sleep_time_duration * 60))  # 多少秒
        start_ind = list(range(0, all_sleep_status.shape[1], dt))
        # 因为最后一个时间段可能不足设定的时间段，所以这里一块返回两者
        values_durations = []
        flys_num = self.roi_flys_nubs
        for i in range(len(start_ind) - 1):
            all_sleep_status_duration = all_sleep_status[:, start_ind[i]:start_ind[i + 1]]
            value = all_sleep_status_duration.sum() / flys_num
            value = value / 60  # 转化为分钟
            sleep_flys_nubs = np.sum(all_sleep_status_duration, axis=-1).astype(np.bool).sum()
            proportion_of_sleep_flys = sleep_flys_nubs / flys_num  # 当前时间段睡觉的果蝇的比例
            values_durations.append([value, dt, proportion_of_sleep_flys])
        last_da = all_sleep_status[:, start_ind[-1]:]
        value = last_da.sum() / flys_num
        value = value / 60  # 转化为分钟
        sleep_flys_nubs = np.sum(last_da, axis=-1).astype(np.bool).sum()
        proportion_of_sleep_flys = sleep_flys_nubs / flys_num  # 当前时间段睡觉的果蝇的比例
        values_durations.append([value, last_da.shape[1], proportion_of_sleep_flys])
        values_durations = np.array(values_durations)
        np.save(str(npy_path), values_durations)
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
        all_datas = self.all_datas.astype(np.float64)
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

    def heatmap_to_pcolor(self, heat, mask):
        """
        转伪彩图
        :return:
        """
        # 尝试了生成16位的伪彩图，发现applyColorMap函数不支持
        max_v, datatype = 255, np.uint8
        heat = equalizeHist_use_mask(heat, mask, notuint8=True)
        heat = heat / heat.max() * max_v
        heat = np.round(heat).astype(datatype)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        return heat

    def PARAM_heatmap(self, p):
        '''
        跟roi没有关系，算的是所有果蝇的热图。
        :param p:
        :return:
        '''
        heatmap = self.heatmap.copy()
        heatmaps = []

        for mask, cp in zip(self.mask_imgs, self.cps):
            hm = heatmap * mask
            pcolor = self.heatmap_to_pcolor(hm, mask)
            pcolor *= np.tile(mask[:, :, None], (1, 1, 3))  # 只有在这mask一下，后面才能叠加
            heatmaps.append(pcolor)

        heatmap_img = np.array(heatmaps).sum(axis=0).astype(np.uint8)  # 叠加后的图像背景是黑的
        mask_all = np.array(self.mask_imgs).sum(axis=0)
        mask_all = (mask_all == 0).astype(np.uint8) * 128  # 背景蓝色 bgr(128,0,0)
        heatmap_img[:, :, 0] += mask_all
        # cv2.imshow('', heatmap_img)
        # cv2.waitKeyEx()
        cv2.imwrite(str(p), heatmap_img)

    def PARAM_heatmap_barycenter(self, p, p_heatmap):
        '''
        计算热图重心，并可视化
        :return:
        '''

        def get_barycenter_of_mat(m):  # 求矩阵重心
            def get_barycenter_of_line(l):  # 求直线重心
                i = np.arange(len(l))
                return np.sum(l * i) / np.sum(l)

            lx = np.sum(m, axis=0)
            ly = np.sum(m, axis=1)
            return (get_barycenter_of_line(lx),
                    get_barycenter_of_line(ly))

        barycps = []
        heatmap = self.heatmap
        r = self.dish_radius
        for cp in self.cps:
            p0 = (cp[0] - r, cp[1] - r)
            m = heatmap[
                p0[1]:p0[1] + 2 * r + 1,
                p0[0]:p0[0] + 2 * r + 1]
            barycp = get_barycenter_of_mat(m)
            barycps.append((barycp[0] + p0[0],
                            barycp[1] + p0[1]))

        self.barycps = barycps

        img = cv2.imread(str(p_heatmap))
        img = np.zeros_like(img)

        def dist2p(p1, p2):
            return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

        barycps_r = []
        for bp, cp in zip(barycps, self.cps):
            dist = dist2p(bp, cp)
            barycps_r.append(dist)
            cv2.circle(img, tuple(cp), self.dish_radius, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.circle(img, tuple(cp), int(round(dist)), (255, 255, 0), -1, cv2.LINE_AA)
            if dist == 0:  # 不画
                continue
            else:
                x = (bp[0] - cp[0]) * self.dish_radius / dist + cp[0]
                y = (bp[1] - cp[1]) * self.dish_radius / dist + cp[1]
                x = int(round(x))
                y = int(round(y))
                # cv2.line(img, tuple(cp), (x, y), (0, 0, 255), 1, cv2.LINE_AA)
                cv2.arrowedLine(img, tuple(cp), (x, y), (0, 0, 255), 1, cv2.LINE_AA)
        # np.save(r'Z:\dataset\qususu\ceshishipin\v080\output_72hole_0330_v080\plot_images\barycps_r.npy', barycps_r)
        # cv2.imshow('', img)
        # cv2.waitKeyEx()
        cv2.imwrite(str(p), img)

    def PARAM_heatmap_of_roi(self, p):
        '''
        根据当前roi组来算热图，组内不管有多少个圆圈，只算一个平均的，并对热图放大显示。
        :return:
        '''
        heatmap = self.heatmap.copy()
        r = self.dish_radius
        heatmap_sum = np.zeros([r * 2 + 1] * 2, dtype=heatmap.dtype)
        for roi_id in self.roi_flys_id:
            x, y = self.cps[roi_id]
            a_hp = heatmap[y - r:y + r + 1, x - r:x + r + 1]
            heatmap_sum += a_hp
        mask = np.zeros(heatmap_sum.shape, np.uint8)
        cv2.circle(mask, (r, r), r, 255, -1)
        mask = mask != 0
        pcolor = self.heatmap_to_pcolor(heatmap_sum, mask)
        # pcolor *= np.tile(mask[:, :, None], (1, 1, 3))
        # pcolor = cv2.resize(pcolor, dsize=None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        pcolor = cv2.resize(pcolor, dsize=None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(p), pcolor)
        # pcolor = cv2.GaussianBlur(pcolor, (5, 5), 0)
        # cv2.imshow('', pcolor)
        # cv2.waitKeyEx()
        # exit()
        # ...

    def PARAM_heatmap_exclude_sleeptime(self, p1, p2):
        '''
        排除睡眠时间，然后重新算一遍热图
        实现逻辑：
            改变self.heatmap的值，然后重新运行计算heatmap的函数
        :return:
        '''
        # 因为下面操作要改变self.heatmap的值，所以这个做个备份，等操作完成再还原回来
        heatmap_bak = self.heatmap.copy()

        sleeptime_heatmap = self._get_sleeptime_heatmap()
        heatmap_exclude_sleeptime = self.heatmap - sleeptime_heatmap
        self.heatmap = heatmap_exclude_sleeptime
        if not p1.exists():
            self.PARAM_heatmap(p1)
        self.PARAM_heatmap_of_roi(p2)

        self.heatmap = heatmap_bak  # 还原回来

    def _get_sleeptime_heatmap(self):
        '''
        计算睡眠时间段果蝇活动区域的heatmap，
        :param self:
        :return:
        '''
        sleeptime_heatmap_path = Path(self.cache_dir, 'heatmap_sleeptime.npy')
        if sleeptime_heatmap_path.exists():
            return np.load(sleeptime_heatmap_path)

        cap = cv2.VideoCapture(str(self.video_path))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        sleeptime_heatmap = np.zeros((h, w), np.int)  # 初始化返回值

        # 先计算哪些时间段是睡眠时间（有果蝇在睡觉）
        sleep_status = np.load(Path(self.cache_dir, 'all_sleep_status.npy'))
        timeline = sleep_status.sum(0).astype(np.bool)
        if timeline.sum() == 0:  # 说明没有果蝇在睡觉，所以这里直接返回零矩阵，后面就不用折腾了
            np.save(sleeptime_heatmap_path, sleeptime_heatmap)
            return sleeptime_heatmap
        timeline = np.concatenate([np.array([False]), timeline, np.array([False])], 0)  # 先在两头加上False
        start_t, end_t = [], []
        for i in range(1, len(timeline)):
            pt, t = timeline[i - 1], timeline[i]
            if pt == False and t == True:
                start_t.append(i - 1)
            elif pt == True and t == False:
                end_t.append(i - 1)
        sleep_durations = list(zip(start_t, end_t))  # [起始秒，终止秒)
        sleep_durations = np.array(sleep_durations) * fps  # [起始帧，终止帧)

        # 逐时间段计算睡觉果蝇热图
        seg_th = 120,  # 分割阈值。注意，这俩值要跟前面分割时保持一致
        background_th = 70,  # 跟背景差的阈值。注意，这俩值要跟前面分割时保持一致
        if self.Undistortion_model_path:
            bg_img_path = Path(self.cache_dir, 'background_image_undistort.bmp')
        else:
            bg_img_path = Path(self.cache_dir, 'background_image.bmp')
        bg = cv2.imread(str(bg_img_path))
        gray_bg_int16 = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.int16)
        undistort = Undistortion(self.Undistortion_model_path)
        mask_imgs = np.load(Path(self.cache_dir, 'mask_imgs.npy')).astype(np.bool)
        mask_all = mask_imgs.sum(0).astype(np.bool)
        sleep_status = np.repeat(sleep_status, fps, axis=-1)
        for du_i, (st, ed) in enumerate(sleep_durations):
            print(f'\nsleep duration: {du_i + 1}/{len(sleep_durations)}')
            status = sleep_status[:, st:ed]
            cap.set(cv2.CAP_PROP_POS_FRAMES, st)
            nub = 0
            pbar = Pbar(total=ed - st)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                sleep_fly_id = np.argwhere(status[:, nub] == True)
                if len(sleep_fly_id) == 0:
                    sleep_fly_id = []
                else:
                    sleep_fly_id = list(np.squeeze(sleep_fly_id, axis=1))
                    mask_sleep = np.zeros_like(mask_all)
                    for sl_id in sleep_fly_id:
                        mask_sleep += mask_imgs[sl_id]  # 只mask睡眠的果蝇圆环
                    frame = undistort.do(frame)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    foreground_mask = np.abs(frame.astype(np.int16) - gray_bg_int16) > background_th
                    frame = frame < seg_th
                    frame *= mask_all
                    frame = frame.astype(np.uint8) * 255 * foreground_mask  # 该值是原始heatmap累加分割区域图
                    frame *= mask_sleep  # 原始累加分割区域图跟睡眠果蝇区域mask相乘
                    sleeptime_heatmap += frame.astype(np.bool).astype(np.int)

                nub += 1
                if nub >= ed - st:
                    break
                pbar.update()
            pbar.close()
        np.save(sleeptime_heatmap_path, sleeptime_heatmap)
        return sleeptime_heatmap

    def _get_move_direction_pre_frame(self):
        '''
        算法：某一帧的运动方向由当前帧跟上一帧的坐标点来确定，第一帧因为没有上一帧，第一帧的运动方向设为跟下一帧相同.
        算的是运动方向，不是果蝇身体朝向，注意区分。
        :return:
        '''
        if self.move_direction_pre_frame_path.exists():
            return np.load(self.move_direction_pre_frame_path)
        else:
            img_h, img_w = self.mask_imgs.shape[1:]
            R = self.all_datas
            R[:, :, 1] = img_h - R[:, :, 1]  # 转换坐标系，从左上转到左下
            R1 = R[:, :-1]  # 要计算的上一帧
            R2 = R[:, 1:]  # 要计算的当前帧，从1开始而不是0
            Diff = R2 - R1  # 跟前一帧的差
            Dx, Dy = Diff[:, :, 0], Diff[:, :, 1]  # 分解为横坐标以及纵坐标的差
            Ds = (Dx ** 2 + Dy ** 2) ** 0.5  # 差组成的直角三角形的斜边边长
            Theta = np.arccos(Dx / Ds)
            Theta = np.where(np.isnan(Theta), 0, Theta)  # 去除因为除零造成的nan值
            Theta = Theta * 180 / np.pi  # 转换成角度值，但是arccos的值域是0-180，咱的期望值域是0-360
            Theta = np.where(Dy < 0, 360 - Theta, Theta)  # Dy小于0的地方角度应该是180-360，而不是0-180，需要拿360减去当前值来修正。
            Theta = np.pad(Theta, ((0, 0), (1, 0)), 'edge')  # 第一帧的运动方向设置为跟下一帧相同
            np.save(self.move_direction_pre_frame_path, Theta)
            return Theta

    def _get_fly_angle_cor(self):
        '''
        根据运动方向来修正果蝇头部朝向
        :return:
        '''
        if self.fly_angles_cor_path.exists():
            return np.load(self.fly_angles_cor_path)
        else:
            move_ang = self._get_move_direction_pre_frame()
            move_ang = np.transpose(move_ang, [1, 0])
            fly_ang = np.load(Path(self.cache_dir, 'fly_angles.npy'))
            diff = np.abs(move_ang - fly_ang)
            mask = (diff > 90) * (diff < 270)
            fly_ang_cor = np.where(mask, fly_ang + 180, fly_ang)
            np.save(self.fly_angles_cor_path, fly_ang_cor)
            return fly_ang_cor

    def PARAM_angle_changes(self):
        # if self.angle_changes_path.exists():
        #     return self.angle_changes_path
        ang = self._get_fly_angle_cor()
        ang_sec = ang[::self.fps, self.roi_flys_id]
        as1 = ang_sec[:-1]
        as2 = ang_sec[1:]
        changes = np.abs(as2 - as1)
        changes = np.where(changes > 180, 360 - changes, changes)  # 相比前一秒的变化角度（0-180）
        # ana_duration_secs = int(self.ana_time_duration * 60)
        ana_duration_secs = int(self.angle_time_duration * 60)
        ana_times = int(len(changes) / ana_duration_secs) + 1
        # 按照时间段来分出来
        changes_es = [changes[i * ana_duration_secs:(i + 1) * ana_duration_secs] for i in range(ana_times)]
        if len(changes_es[-1]) < len(changes_es[-2]) * 0.1:  # 最后一段太小的话就舍弃
            changes_es = changes_es[:-1]
        bins = 18  # 直方图横坐标维度
        hists = []
        '''
        这里注意一下，求出来的直方图横坐标是0-10，10-20,20-30，。。。，170-180，更具体来说，应该是这样：
        [0,10),[10,20),[20,30),...[170,180]，前闭后开，但是最后一个是全闭。加上0后变为：
        0，(0,10),[10,20),[20,30),...[170,180]
        '''
        zeros_nums = []
        for cha in changes_es:
            hist = np.histogram(cha.flatten(), bins=bins, range=(0, 180))
            hists.append(hist)
            zeros_nums.append(np.sum(cha == 0))
        # np.save(self.angle_changes_path, hists)
        return hists, zeros_nums


if __name__ == '__main__':
    da = np.array([1, 1, 2, 3, 9, 7, 4, 8, 19, 20])
    print(np.histogram(da, bins=5, range=(0, 20)))
