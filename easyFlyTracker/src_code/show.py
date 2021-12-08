#!/GPFS/zhangli_lab_permanent/zhuqingjie/env/py3_tf2/bin/python
'''
@Time    : 20/07/21 下午 02:03
@Author  : zhuqingjie 
@User    : zhu
@FileName: show.py
@Software: PyCharm
'''
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import pickle
from matplotlib.font_manager import FontProperties

matplotlib.use('agg')
import matplotlib.pyplot as plt
from easyFlyTracker.src_code.analysis import Analysis
from easyFlyTracker.src_code.utils import args_filter, Wait


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
            AB_dist_mm,
            # scale,  # 比例尺，毫米/像素
            suffix='all',  # 保存的图片结果后缀
            roi_flys_ids=None,
            heatmap_remove_sleep=False,  # 是否计算抠除睡眠后的heatmap
            isfirst=True,  # 通常不同roi组会计算多次，但是有的结果只需计算一次，为避免重复计算这里用来标识是否是第一次
            ana_time_duration=None,  # 主要是为了生成的图片命名的时候用到
    ):
        '''

        :param suffix: 保存的图片或者npy文件的后缀，用于区分不同统计方式的结果
        '''
        # 初始化各种文件夹
        self.video_path = Path(video_path)
        self.output_dir = output_dir
        self.saved_dir = Path(output_dir, 'plot_images')
        self.saved_dir_excels = Path(output_dir, 'plot_excels')
        self.saved_dir_npys = Path(self.saved_dir, '.npys')
        self.saved_dir.mkdir(exist_ok=True)
        self.saved_dir_excels.mkdir(exist_ok=True)
        self.saved_dir_npys.mkdir(exist_ok=True)

        self.video_stem = Path(video_path).stem
        cap = cv2.VideoCapture(str(self.video_path))
        self.fps = round(cap.get(cv2.CAP_PROP_FPS))  # 从opencv得到的帧率不太准，实际是30，opencv给的却是29.99，对后续计算有影响，所以取整
        cap.release()
        self.saved_suffix = suffix
        self.isfirst = isfirst
        self.ana_time_duration = ana_time_duration

        # roi_flys
        res = np.load(Path(self.output_dir, '.cache', 'track.npy'))
        if roi_flys_ids == None:
            self.roi_flys_list = np.array([True] * len(res))
        else:
            self.roi_flys_list = np.array([False] * len(res))
            self.roi_flys_list[roi_flys_ids] = True
        self.roi_flys_id = [i for i, r in enumerate(self.roi_flys_list) if r]
        self.heatmap_remove_sleep = heatmap_remove_sleep

        # 计算比例尺
        # config_pk = pickle.load(open(Path(self.output_dir, 'config.pkl'), 'rb'))
        # config_pk = np.array(config_pk)
        # # self.cps = config_pk[:, :2]
        # self.dish_radius = int(round(float(np.mean(config_pk[:, -1]))))
        pklf = Path(output_dir, 'config.pkl')
        _, AB_dist = pickle.load(open(pklf, 'rb'))
        self.sacle = AB_dist_mm / AB_dist
        # print(f'scale: {self.sacle}')
        # print('请输入所选两点之间的实际距离，单位毫米：')
        # dist_mm = float(input())
        # print('请输入所选两点之间的像素距离，单位像素：')
        # dist_piexl = float(input())
        # self.sacle = dist_mm / dist_piexl
        # print(f'scale: {self.sacle}')

        # 获取视频对应的Analysis实例
        self.ana = Analysis(**ana_params)

        # 设置字体格式
        self.font_times = FontProperties(fname=str(Path(Path(__file__).parent.parent, 'fonts/times.ttf')), size=12)
        self.font_timesbd = FontProperties(fname=str(Path(Path(__file__).parent.parent, 'fonts/timesbd.ttf')), size=15)

    def SHOW_avg_dist_per10min(self):
        '''
        十分钟一个值，
        :return:
        '''
        datas_paths = self.ana.PARAM_speed_displacement(redo=True)  # 不调缓存，重新计算
        datas = np.load(datas_paths[1])
        datas *= self.sacle
        xs = list(range(1, len(datas) + 1))
        xs = [str(_) for _ in xs]
        plt.close()
        plt.rcParams['figure.figsize'] = (15.0, 8.0)
        plt.grid(linewidth=1)
        plt.xlabel(f'Video Time (per {self.ana_time_duration} mins)', fontproperties=self.font_timesbd)
        plt.ylabel('Distances (mm)', fontproperties=self.font_timesbd)
        plt.title('Average distances per duration at different time', fontproperties=self.font_timesbd)
        plt.plot(xs, datas, label=self.video_stem)
        plt.scatter(xs, datas)
        plt.xticks(fontproperties=self.font_times)
        plt.yticks(fontproperties=self.font_times)
        plt.legend(prop={'family': 'Times New Roman', 'size': 12})
        # sns.lineplot(x='x', y='y', data={'x': xs, 'y': datas}) Video Time (per 10 mins)
        # plt.show()  average_distances_per_flies_per_x_mins_merge.png
        plt.savefig(str(Path(self.saved_dir,
                             f'average_distances_per_flies_per_{self.ana_time_duration}_mins_[{self.saved_suffix}].png')))
        np.save(str(Path(self.saved_dir_npys,
                         f'average_distances_per_flies_per_{self.ana_time_duration}_mins_[{self.saved_suffix}].npy')),
                datas)
        df = pd.DataFrame(data=datas, columns=[self.saved_suffix])
        df.to_excel(Path(self.saved_dir_excels,
                         f'average_distances_per_flies_per_{self.ana_time_duration}_mins_[{self.saved_suffix}].xlsx'))

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
        df.to_excel(Path(self.saved_dir_excels, f'dist_change_per_h_{self.saved_suffix}.xlsx'))

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
        df.to_excel(Path(self.saved_dir_excels, f'in_centre_prob_per_h_{self.saved_suffix}.xlsx'))

    def SHOW_sleep_time_per_h(self):
        '''
        每个时间段睡眠总时间/有效果蝇数
        单位：分钟
        :return:
        '''
        datas_path = self.ana.PARAM_sleep_status(redo=True)
        da = np.load(datas_path)
        # duration_times是对应时段持续时间，最后一个不一定跟前面相等
        da, duration_times, proportion_of_sleep_flys = da[:, 0], da[:, 1], da[:, 2]
        xs = list(range(1, len(da) + 1))
        xs = [str(_) for _ in xs]

        plt.close()
        plt.rcParams['figure.figsize'] = (15.0, 8.0)
        plt.grid(linewidth=1)

        # 绘制折线图
        plt.xlabel(f'Video Time (per {self.ana.sleep_time_duration} mins)', fontproperties=self.font_timesbd)
        plt.ylabel('Sleep (min) ', fontproperties=self.font_timesbd)
        plt.title('Average sleep time per flies per duration & Proportion of sleep flies',
                  fontproperties=self.font_timesbd)
        plt.plot(xs, da, label=f'{self.video_stem}, Sleep')
        plt.scatter(xs, da)
        plt.ylim([0, da.max() * 1.2])
        plt.xticks(fontproperties=self.font_times)
        plt.yticks(fontproperties=self.font_times)
        plt.legend(loc='upper left', frameon=True, prop={'family': 'Times New Roman', 'size': 12})

        # 绘制柱状图
        ax2 = plt.twinx()
        ax2.bar(xs, proportion_of_sleep_flys, label=f'{self.video_stem}, Proportion of sleep flies',
                width=0.3, alpha=0.2)
        ax2.set_ylim([0, 1])
        ax2.set_ylabel('Proportion of sleep flies', fontproperties=self.font_timesbd)
        ax2.legend(loc='upper right', frameon=True, prop={'family': 'Times New Roman', 'size': 12})
        plt.yticks(fontproperties=self.font_times)

        # plt.show()
        plt.savefig(str(Path(self.saved_dir,
                             f'average_sleep_time_per_{self.ana.sleep_time_duration}_mins_&_proportion_of_sleep_flies_[{self.saved_suffix}].png')))
        da = [da, proportion_of_sleep_flys]
        np.save(str(Path(self.saved_dir_npys,
                         f'average_sleep_time_per_{self.ana.sleep_time_duration}_mins_&_proportion_of_sleep_flies_[{self.saved_suffix}].npy')),
                da)
        df = pd.DataFrame(data=da,
                          index=[f'average_sleep_time_per_{self.ana.sleep_time_duration}_mins',
                                 'proportion_of_sleep_flies'],
                          columns=[str(_ + 1) for _ in range(len(da[0]))])
        df.to_excel(Path(self.saved_dir_excels,
                         f'average_sleep_time_per_{self.ana.sleep_time_duration}_mins_&_proportion_of_sleep_flies_[{self.saved_suffix}].xlsx'))

    def SHOW_heatmap(self):
        '''
        heatmap只有一张，针对不同的roi算不同的heatmap没有意义，所以只算一次总的。
        :return:
        '''
        p = Path(self.saved_dir, f'frequency_heatmap_per_flies.png')
        if self.isfirst:  # 若已存在不再重复计算
            self.ana.PARAM_heatmap(p)

    def SHOW_heatmap_of_roi(self):
        '''
        每批的roi组只算一个热图，并且放大处理。
        :return:
        '''
        p = Path(self.saved_dir, f'frequency_heatmap_GroupAverage_[{self.saved_suffix}].png')
        self.ana.PARAM_heatmap_of_roi(p)

    def SHOW_heatmap_barycenter(self):
        '''
        根据heatmap计算重心，并画出来
        :return:
        '''
        p_heatmap = Path(self.saved_dir, f'frequency_heatmap_per_flies.png')
        p = Path(self.saved_dir, f'frequency_location_change_per_flies.png')
        if self.isfirst:  # 若已存在不再重复计算
            self.ana.PARAM_heatmap_barycenter(p, p_heatmap)

            # 保存excel
            barycps, cps = self.ana.barycps, self.ana.cps
            barycps = np.array(barycps)
            data = np.concatenate([cps, barycps], axis=1)
            df = pd.DataFrame(data=data, index=list(range(len(data))),
                              columns=['centre-x', 'centre-y', 'barycenter-x', 'barycenter-y'])
            df.to_excel(Path(self.saved_dir_excels, f'frequency_location_change_per_flies.xlsx'))
            ...

    def SHOW_heatmap_exclude_sleeptime(self):
        '''
        展示去除睡眠时间的果蝇活动区域热图
        :return:
        '''
        p1 = Path(self.saved_dir, f'heatmap_exclude_sleeptime.png')
        p2 = Path(self.saved_dir, f'heatmap_exclude_sleeptime_[{self.saved_suffix}].png')
        self.ana.PARAM_heatmap_exclude_sleeptime(p1, p2)

    def SHOW_angle_changes(self):
        hists, zeros_nums = self.ana.PARAM_angle_changes()
        edge = hists[0][1]
        hists = np.array([h[0] for h in hists])
        xs = [f'{int(edge[i])}-{int(edge[i + 1])}' for i in range(len(edge) - 1)]
        columns = [f'[{int(edge[i])},{int(edge[i + 1])})' for i in range(len(edge) - 1)]
        '''
        这里注意一下，横坐标是0-10，10-20,20-30，。。。，170-180，更具体来说，应该是这样
        [0,10),[10,20),[20,30),...[170,180]，前闭后开，但是最后一个是全闭.加上0后变为：
        0，(0,10),[10,20),[20,30),...[170,180]
        '''
        # 把0元素个数加进去需要做的事，1，横坐标加个0；2，原来0-10的元素个数减去0的个数
        xs = ['0'] + xs  # 把零加进去
        columns = ['0'] + columns
        columns[1] = '(' + columns[1][1:]
        columns[-1] = columns[-1][:-1] + ']'
        zeros_nums = np.array(zeros_nums)[:, None]
        hists = np.concatenate((zeros_nums, hists), axis=1)
        hists[:, 1] -= hists[:, 0]

        plt.close()
        plt.rcParams['figure.figsize'] = (15.0, 8.0)
        plt.grid(linewidth=1)
        plt.xlabel('Angle region (degree)', fontproperties=self.font_timesbd)
        plt.ylabel('Frequency (times)', fontproperties=self.font_timesbd)
        plt.title('Histogram of angle change per duration', fontproperties=self.font_timesbd)
        for i, hi in enumerate(hists):
            plt.plot(xs, hi, label=f'Duration {i + 1}')
            plt.scatter(xs, hi)
        plt.xticks(fontproperties=self.font_times)
        plt.yticks(fontproperties=self.font_times)
        plt.legend(prop={'family': 'Times New Roman', 'size': 12})
        plt.legend(loc='upper right')
        plt.savefig(str(Path(self.saved_dir, f'angle_change_per_duration_[{self.saved_suffix}].png')))
        np.save(str(Path(self.saved_dir_npys, f'angle_change_per_duration_[{self.saved_suffix}].npy')), hists)
        df = pd.DataFrame(data=hists, columns=columns, index=[f'Duration {i + 1}' for i in range(len(hists))])
        df.to_excel(Path(self.saved_dir_excels, f'angle_change_per_duration_[{self.saved_suffix}].xlsx'))
        ...

    def show_all(self):
        self.SHOW_avg_dist_per10min()
        # self.SHOW_dist_change_per_h()
        # self.SHOW_in_centre_prob_per_h()
        self.SHOW_sleep_time_per_h()
        self.SHOW_heatmap()
        self.SHOW_heatmap_of_roi()
        self.SHOW_heatmap_barycenter()
        if self.heatmap_remove_sleep:
            self.SHOW_heatmap_exclude_sleeptime()
        self.SHOW_angle_changes()


def merge_sleep_time_result(params):
    suffixs = [v[-1] for v in params['rois']]
    sleep_time_duration = params['sleep_time_duration']
    font_times = FontProperties(fname=str(Path(Path(__file__).parent.parent, 'fonts/times.ttf')), size=12)
    font_timesbd = FontProperties(fname=str(Path(Path(__file__).parent.parent, 'fonts/timesbd.ttf')), size=15)
    dir = Path(params['output_dir'], 'plot_images', '.npys')
    dst_dir = Path(params['output_dir'], 'plot_images')
    excel_dir = Path(params['output_dir'], 'plot_excels')
    das = [
        np.load(Path(dir, f'average_sleep_time_per_{sleep_time_duration}_mins_&_proportion_of_sleep_flies_[{s}].npy'))
        for s in suffixs]
    xs = list(range(1, len(das[0][0]) + 1))
    xs = [str(_) for _ in xs]

    plt.close()
    plt.rcParams['figure.figsize'] = (15.0, 8.0)
    plt.grid(linewidth=1)

    # 绘制折线图
    plt.xlabel(f'Video Time (per {sleep_time_duration} mins)', fontproperties=font_timesbd)
    plt.ylabel('Sleep (min) ', fontproperties=font_timesbd)
    plt.title('Average sleep time per flies per duration & Proportion of sleep flies',
              fontproperties=font_timesbd)
    for da, suf in zip(das, suffixs):
        plt.plot(xs, da[0], label=f'{suf}, Sleep')
        plt.scatter(xs, da[0])
    plt.ylim([0, max([da[0].max() for da in das]) * 1.2])
    plt.xticks(fontproperties=font_times)
    plt.yticks(fontproperties=font_times)
    plt.legend(loc='upper left', frameon=True, prop={'family': 'Times New Roman', 'size': 12})

    # 绘制柱状图
    total_width = 0.5
    n = len(das)
    width = total_width / n
    xs_ori = np.array(list(range(len(das[0][0])))) - (total_width - width) / 2
    ax2 = plt.twinx()
    for en, (da, suf) in enumerate(zip(das, suffixs)):
        ax2.bar(xs_ori + width * en, da[1], label=f'{suf}, Proportion of sleep flies', width=width, alpha=0.2)
    ax2.set_ylim([0, 1])
    ax2.set_ylabel('Proportion of sleep flies', fontproperties=font_timesbd)
    ax2.legend(loc='upper right', frameon=True, prop={'family': 'Times New Roman', 'size': 12})
    plt.yticks(fontproperties=font_times)

    # plt.show()
    plt.savefig(str(
        Path(dst_dir, f'average_sleep_time_per_{sleep_time_duration}_mins_&_proportion_of_sleep_flies_[merge].png')))
    np.save(str(
        Path(dir, f'average_sleep_time_per_{sleep_time_duration}_mins_&_proportion_of_sleep_flies_[merge].npy')),
        das)
    df = pd.DataFrame(data=np.array([da[0] for da in das]),
                      index=suffixs,
                      columns=[str(_ + 1) for _ in range(len(das[0][0]))])
    df.to_excel(Path(excel_dir, f'average_sleep_time_per_{sleep_time_duration}_mins_[merge].xlsx'))
    df = pd.DataFrame(data=np.array([da[1] for da in das]),
                      index=suffixs,
                      columns=[str(_ + 1) for _ in range(len(das[0][0]))])
    df.to_excel(Path(excel_dir, f'proportion_of_sleep_flies_{sleep_time_duration}_mins_[merge].xlsx'))


def merge_result(params):
    suffixs = [v[-1] for v in params['rois']]
    ana_time_duration = params['ana_time_duration']
    sleep_time_duration = params['sleep_time_duration']
    prefixs = [  # 前缀、x轴标签、y轴标签，title
        [f'average_distances_per_flies_per_{ana_time_duration}_mins',
         f'Video Time (per {ana_time_duration} mins)',
         'Distances (mm)',
         'Average distances per duration at different time'],
        [f'average_sleep_time_per_{sleep_time_duration}_mins',
         f'Video Time (per {sleep_time_duration} mins)',
         'Sleep (min) ',
         'Average sleep time per flies per duration'],
    ]
    font_times = FontProperties(fname=str(Path(Path(__file__).parent.parent, 'fonts/times.ttf')), size=12)
    font_timesbd = FontProperties(fname=str(Path(Path(__file__).parent.parent, 'fonts/timesbd.ttf')), size=15)
    dir = Path(params['output_dir'], 'plot_images', '.npys')
    dst_dir = Path(params['output_dir'], 'plot_images')
    for pre, xl, yl, title in prefixs:
        if 'average_sleep_time_per' in pre:  # 这个单独处理
            merge_sleep_time_result(params)
            continue
        plt.close()
        plt.rcParams['figure.figsize'] = (15.0, 8.0)
        plt.grid(linewidth=1)
        plt.xlabel(xl, fontproperties=font_timesbd)
        plt.ylabel(yl, fontproperties=font_timesbd)
        plt.title(title, fontproperties=font_timesbd)
        das = [np.load(Path(dir, f'{pre}_[{suf}].npy')) for suf in suffixs]
        for da, lb in zip(das, suffixs):
            da = np.squeeze(da)
            xs = list(range(1, len(da) + 1))
            xs = [str(_) for _ in xs]
            plt.plot(xs, da, label=' ' + str(lb))
            plt.scatter(xs, da)
        plt.xticks(fontproperties=font_times)
        plt.yticks(fontproperties=font_times)
        plt.legend(prop={'family': 'Times New Roman', 'size': 12})
        plt.savefig(str(Path(dst_dir, f'{pre}_[merge].png')))
        np.save(str(Path(dir, f'{pre}_[merge].npy')), das)
        ...  # 顺便保存个merge的excel
        df = pd.DataFrame(np.array(das).T, columns=suffixs)
        df.to_excel(Path(params['output_dir'], 'plot_excels', f'{pre}_[merge].xlsx'))


def one_step_run(params):
    '''
    Show类只针对一次roi的运算，多个区域的roi运算要分多次运行
    :param params:
    :return:
    '''
    rois = params['rois']

    # Analysis需要传入一个参数sleep_dist_th_per_second，先在这里计算出来
    # 为啥呢？因为Analysis所有的计算都是基于像素单位，而不同比例尺下上面参数的值不应该相同，所以要先算出该值。
    # 该参数不暴露给用户，在此定死，为下值，单位毫米：
    sleep_dist_th_per_second_mm = 1.5
    AB_dist_mm = params['AB_dist_mm']
    cp = Path(params['output_dir'], 'config.pkl')
    with open(cp, 'rb') as f:
        AB_dist = pickle.load(f)[1]
    pixel_per_mm = AB_dist / AB_dist_mm
    sleep_dist_th_per_second = sleep_dist_th_per_second_mm * pixel_per_mm
    sleep_dist_th_per_second = int(round(sleep_dist_th_per_second))

    for notfirst, (ids, flag) in enumerate(rois):
        ana_params = args_filter(params, Analysis)
        ana_params['roi_flys_flag'] = flag
        ana_params['roi_flys_ids'] = ids
        ana_params['sleep_dist_th_per_second'] = sleep_dist_th_per_second
        show_params = args_filter(params, Show)
        show_params['roi_flys_ids'] = ids
        show_params['suffix'] = flag
        show_params['ana_params'] = ana_params
        show_params['isfirst'] = not notfirst
        print('-' * 50)
        print(f'Group name: {flag}')
        print(f'Group ids : {ids}')
        with Wait():
            s = Show(**show_params)
            s.show_all()

    if len(rois) > 1:
        merge_result(params)


if __name__ == '__main__':
    p = r'D:\Pycharm_Projects\qu_holmes_su_release\tests\output2\config.pkl'
    with open(p, 'rb') as f:
        data = pickle.load(f)
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
