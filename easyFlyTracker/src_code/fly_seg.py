#!/GPFS/zhangli_lab_permanent/zhuqingjie/env/py3_tf2/bin/python
'''
@Time    : 20/07/21 下午 06:06
@Author  : zhuqingjie 
@User    : zhu
@FileName: fly_seg.py
@Software: PyCharm
'''
import numpy as np
import cv2, time, random, math, cv2_ext
from pathlib import Path
from scipy import stats
from multiprocessing import Pool
from threading import Thread
from easyFlyTracker.src_code.utils import Pbar, Wait
from easyFlyTracker.src_code.Camera_Calibration import Undistortion
from easyFlyTracker.src_code.utils import stop_thread
from easyFlyTracker.src_code.gui_config import GUI_CFG


class FlySeg():
    '''
        计算视频中每个果蝇的运动轨迹
        返回果蝇的坐标，当下列情况发生时返回异常坐标(-1,-1)：
        1，该果蝇不属于roi果蝇；  ---> 该果蝇所有帧结果都为(-1,-1)
        2，mask二值图像的连通区域小于2个（也就是说该帧果蝇分割失败）   ---> 该果蝇个别帧为(-1,-1)
        '''
    __doc__ = 'flyseg'

    def __init__(
            self,
            video_path,  # 视频路径
            output_dir,  # 输出文件夹
            save_txt_name,  # 要保存的txt name（同时保存txt和同名npy）,不要求绝对路径，只要求name即可
            begin_time,  # 从哪个时间点开始
            # h_num, w_num,  # 盘子是几乘几的
            Undistortion_model_path=None,  # 畸变矫正参数路径
            duration_time=None,  # 要持续多长时间
            # dish_exclude=None,  # 排除的特殊圆盘，比如空盘、死果蝇等情况,可以一维或者（h_num, w_num），被排除的圆盘结果用(-1,-1)表示
            seg_th=120,  # 分割阈值
            background_th=70,  # 跟背景差的阈值
            area_th=0.5,  # 内圈面积阈值
            # minR_maxR_minD=(40, 50, 90),  # 霍夫检测圆时的参数，最小半径，最大半径，最小距离
            skip_config=False,
    ):
        # 初始化各种文件夹
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.saved_dir = Path(self.output_dir, '.cache')  # 中间结果文件夹
        # 因为加畸变矫正跟不加畸变矫正背景图像不一样，所以用两个文件名来区分
        if Undistortion_model_path:
            self.bg_img_path = Path(self.saved_dir, 'background_image_undistort.bmp')
        else:
            self.bg_img_path = Path(self.saved_dir, 'background_image.bmp')
        self.res_txt_path = Path(self.output_dir, save_txt_name)  # txt结果给用户看，所以保存到用户文件夹
        self.res_npy_path = Path(self.saved_dir, f'{save_txt_name[:-3]}npy')
        self.saved_dir.mkdir(exist_ok=True)

        self.video_stem = str(Path(video_path).stem)
        self.seg_th = seg_th
        self.undistort = Undistortion(Undistortion_model_path)
        self.background_th = background_th

        self.video = cv2.VideoCapture(str(self.video_path))
        self.video_frames_num = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = round(self.video.get(cv2.CAP_PROP_FPS))

        # gui config
        _, temp_frame = self.video.read()

        # 在这判断训练畸变矫正模型所使用的图像分辨率是否跟当前视频一致，前提是加畸变矫正
        if Undistortion_model_path:
            map_sp = self.undistort.mapxy.shape[-2:]
            frame_sp = temp_frame.shape[:2]
            if map_sp != frame_sp:
                print('The resolution of training calibration_model images is not same as the resolution of video!')
                exit()
        # 如果跳过config，那么必须有config.pkl文件
        if skip_config:
            if not Path(self.output_dir, 'config.pkl').exists():
                print("'config.pkl' file is not exists!")
                exit()
        temp_frame = self.undistort.do(temp_frame)
        g = GUI_CFG(temp_frame, [], str(self.output_dir))
        res = g.CFG_circle(direct_get_res=skip_config)
        if len(res) == 0: raise ValueError
        rs = [re[-1] for re in res]
        self.dish_radius = int(round(float(np.mean(np.array(rs)))))
        self.region_radius = int(round(math.sqrt(area_th) * self.dish_radius))
        self.cps = [tuple(re[:2]) for re in res]

        # get rois and mask images
        self._get_rois()
        self._get_maskimgs()

        # 计算背景
        self.comp_bg()

        # set begin frame
        begin_frame = round(begin_time * 60 * self.video_fps)
        self.begin_frame = begin_frame
        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.begin_frame)

        if duration_time in (None, 'None', 0):
            self.duration_frames = self.video_frames_num - self.begin_frame
        else:
            self.duration_frames = duration_time * 60 * self.video_fps

        # 如果用户配置不合理，设置的时间点超出了视频时长，则按照真实视频时长来截取
        if self.duration_frames > self.video_frames_num:
            self.duration_frames = self.video_frames_num

    def _get_rois(self):
        r = self.dish_radius
        # (h_start, h_end, w_start, w_end)
        self.rois = [
            (cp[1] - r, cp[1] + r, cp[0] - r, cp[0] + r)
            for cp in self.cps
        ]

    def _get_maskimgs(self):
        h = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        # h, w = 485,863
        self.mask_imgs = [
            cv2.circle(np.zeros((h, w), np.uint8), cp, self.dish_radius, 255, -1)
            for cp in self.cps
        ]
        mask_all = np.zeros((h, w), np.bool)
        for img in self.mask_imgs:
            mask_all += img.astype(np.bool)
        # mask_all = mask_all.astype(np.uint8) * 255
        self.mask_all = mask_all

    def comp_bg(self):
        # params
        frames_num_used = 800

        if self.bg_img_path.exists():
            bg = cv2.imread(str(self.bg_img_path))
        else:
            with Wait('Collect frames'):
                tim = time.time()
                inds = list(range(self.video_frames_num))
                random.shuffle(inds)
                inds = inds[:frames_num_used]
                frames = []
                for i in inds:
                    # print(f'{i}')
                    self.video.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = self.video.read()
                    frame = self.undistort.do(frame)
                    if ret == False:
                        break
                    frames.append(frame)
                frames = np.array(frames)
            # print(frames.shape)
            with Wait('Calculate the background image'):
                sx = stats.mode(frames)
                bg = sx[0][0]
                bg = cv2.medianBlur(bg, 3)
                cv2.imwrite(str(self.bg_img_path), bg)
            print(f'Finished, time consuming:{time.time() - tim}s')
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.bg = bg
        self.gray_bg_int16 = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.int16)

    def play(self, just_save_one_frame=False):
        i = 0
        pbar = Pbar(total=self.duration_frames)
        while True:
            ret, frame = self.video.read()
            if not ret: break
            frame = self.undistort.do(frame)
            src = frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            foreground_mask = np.abs(frame.astype(np.int16) - self.gray_bg_int16) > self.background_th
            # cv2.imshow('foreground_mask', foreground_mask.astype(np.uint8) * 255)
            frame = frame < self.seg_th
            frame *= self.mask_all
            frame = frame.astype(np.uint8) * 255 * foreground_mask
            for cp in self.cps:
                cv2.circle(frame, cp, self.dish_radius, 255, 1)
                cv2.circle(src, cp, self.dish_radius, (0, 0, 255), 1)
                # cv2.circle(frame, cp, cfg.Region.radius, 175, 1)
                # cv2.circle(src, cp, cfg.Region.radius, 200, 1)
            if just_save_one_frame:
                cv2.imwrite(str(Path(self.saved_dir, f'{self.video_stem}_1_mask.bmp')), frame)
                cv2.imwrite(str(Path(self.saved_dir, f'{self.video_stem}_1_src.bmp')), src)
                return

            cv2.imshow('mask', frame)
            cv2.imshow('src', src)
            cv2.waitKey(3)
            i += 1
            pbar.update(1)
            if i >= self.duration_frames:
                pbar.close()
                break
        pbar.close()

    def play_and_show_trackingpoints(self, just_save_one_frame=False):
        res = np.load(self.res_npy_path)

        i = 0
        print('showing...')
        print('q: exit')
        pbar = Pbar(total=self.duration_frames)
        while True:
            ret, frame = self.video.read()
            if not ret:
                pbar.close()
                break
            frame = self.undistort.do(frame)
            for cp, tp in zip(self.cps, res[i]):
                cv2.circle(frame, cp, self.dish_radius, (255, 0, 0), 1)
                cv2.circle(frame, cp, self.region_radius, (0, 255, 0), 1)
                tp = (int(round(tp[0])), int(round(tp[1])))
                # cv2.circle(frame, tp, 3, (0, 0, 255), -1)
                cv2.line(frame, (tp[0] - 10, tp[1]), (tp[0] + 10, tp[1]), (0, 0, 255), 1)
                cv2.line(frame, (tp[0], tp[1] - 10), (tp[0], tp[1] + 10), (0, 0, 255), 1)
            if just_save_one_frame:
                cv2.imwrite(str(Path(self.saved_dir, f'{self.video_stem}_3_frame.bmp')), frame)
                return
            cv2.imshow('frame', frame)
            k = cv2.waitKey(3) & 0xFF
            if chr(k) == 'q' or chr(k) == 'Q':
                break
            i += 1
            pbar.update(1)
            if i >= self.duration_frames:
                pbar.close()
                break
        # pbar.close()

    def run(self):
        self.fly_centroids = []
        pbar = Pbar(total=self.duration_frames)
        i = 0
        print('tracking...')
        while True:
            ret, frame = self.video.read()
            if not ret:
                break
            frame = self.undistort.do(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            foreground_mask = np.abs(frame.astype(np.int16) - self.gray_bg_int16) > self.background_th
            frame = frame < self.seg_th
            frame *= self.mask_all
            frame = frame.astype(np.uint8) * 255 * foreground_mask
            oneframe_centroids = []
            for roi in self.rois:
                img = frame[roi[0]:roi[1], roi[2]:roi[3]]
                retval, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
                if retval < 2:
                    cent = (-1, -1)
                else:
                    cent = centroids[np.argmax(stats[1:, -1]) + 1]
                    cent = (round(cent[0] + roi[2], 2),
                            round(cent[1] + roi[0], 2))
                oneframe_centroids.append(cent)
            self.fly_centroids.append(oneframe_centroids)
            i += 1
            pbar.update()
            if i >= self.duration_frames:
                break
        pbar.close()
        self._save()
        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.begin_frame)

    def _save(self):
        # 考虑到发布版本一次运行所保存的单个文件比较大，所以这里不再保存txt仅保存npy文件
        with open(self.res_txt_path, 'w') as f:
            # 由于计算出来的begin_frame点可能跟上次计算的结果有重复，导致所有结果相加长度不等于总帧数，所以在此保存一下每次结果的起始点
            f.write(f'{self.begin_frame}\n')
            for line in self.fly_centroids:
                f.write(f'{line}\n')
        np.save(self.res_npy_path, np.array(self.fly_centroids, dtype=np.float64))


def pbarFilenubs(dir, total, fmt='*.npy'):
    pbar = Pbar(total=total)
    d = Path(dir)
    while True:
        if d.exists():
            filenub = len(list(d.rglob(fmt)))
        else:
            filenub = 0
        pbar.update(set=True, set_value=filenub)
        time.sleep(0.2)


def fn(params):
    s = FlySeg(**params)
    s.run()


def multiprocessing(seg_params, cpus=45):
    cap = cv2.VideoCapture(seg_params['video_path'])
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    time = frames / fps / 60

    params = []
    for t in range(0, round(time), seg_params['duration_time']):
        params.append({**seg_params, 'begin_time': t, 'save_txt_name': f'{t:0>4d}.txt'})
    FlySeg(**params[0])  # 先初始化一下，计算一下背景图和中心点，后面多进程的时候就不用每个都计算了

    print(f'total length: {len(params)}')
    kwargs = {
        'dir': Path(Path(seg_params['video_path']).parent, Path(seg_params['video_path']).stem),
        'total': len(params),
        'fmt': '*.npy'
    }
    thr = Thread(target=pbarFilenubs, kwargs=kwargs)
    thr.start()
    pool = Pool(cpus)
    pool.map(fn, params)
    stop_thread(thr)
    print('done')


def run(cf, mode, just_save_one_frame=True):
    args = ['video_path', 'h_num', 'w_num', 'duration_time', 'seg_th', 'background_th',
            'area_th', 'minR_maxR_minD', 'dish_exclude', 'Undistortion_model_path']
    seg_params = {arg: cf[arg] for arg in args}
    seg_params_play = {
        **seg_params,
        'save_txt_name': '0.txt',
        'begin_time': 150,
    }
    if mode == 1:
        s = FlySeg(**seg_params_play)
        s.play(just_save_one_frame=just_save_one_frame)
        # s.run()
    elif mode == 2:
        t1 = time.time()
        multiprocessing(seg_params, cpus=cf['cpus'])
        print(f'time_used: {(time.time() - t1) / 60} minutes')
    elif mode == 3:
        s = FlySeg(**seg_params_play)
        s.play_and_show_trackingpoints(just_save_one_frame=just_save_one_frame)


if __name__ == '__main__':
    f = FlySeg(
        video_path=r'D:\Pycharm_Projects\qu_holmes_su_release\tests\demo.mp4',
        save_txt_name='0000.txt',
        begin_time=0,
        duration_time=1,
        # config_it=False,
    )
    f.run()
    f.play_and_show_trackingpoints()
