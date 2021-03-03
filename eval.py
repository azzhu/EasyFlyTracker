#! python
# @Time    : 21/03/02 下午 01:46
# @Author  : azzhu 
# @FileName: eval.py
# @Software: PyCharm
import cv2
import pickle
import random
import numpy as np
from pathlib import Path


def 生成若干张标记坐标的图像给专家评估准确性():
    # params
    images_nub = 100
    video_path = r'Z:\dataset\qususu\year_2020\1204\202012041013.avi'
    result_dir = Path(r'Z:\dataset\qususu\year_2020\1204\output')

    cfgfile = Path(result_dir, 'config.pkl')
    with open(cfgfile, 'rb') as f:
        cfg = pickle.load(f)
    cps = np.array(cfg[0])[:, :2]
    mapxy = np.load(r'Z:\dataset\qususu\year_2020\1117\mapx_y.npy')
    output_dir = Path(result_dir, 'eval')
    output_dir.mkdir(exist_ok=True)
    npyfile = Path(result_dir, '.cache', 'track_cor.npy')
    res = np.load(npyfile)
    cap = cv2.VideoCapture(video_path)
    frames_nub = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    random_list = list(range(int(frames_nub)))
    random.shuffle(random_list)
    ids = random.choices(random_list, k=images_nub)

    for i, img_id in enumerate(ids):
        print(i)
        cap.set(cv2.CAP_PROP_POS_FRAMES, img_id)
        _, img = cap.read()
        img = cv2.remap(img, *mapxy, cv2.INTER_LINEAR)
        re = res[:, img_id]
        re = np.round(re).astype(np.int)
        for k, ((x, y), (cx, cy)) in enumerate(zip(re, cps)):
            cv2.putText(img, str(k), (cx + 20, cy - 20), 1, 1, (0, 0, 0))
            img[y, x - 5:x + 6] = [0, 0, 255]
            img[y - 5:y + 6, x] = [255, 0, 0]
        # cv2.imshow('', img)
        # cv2.waitKeyEx()
        cv2.imwrite(str(Path(output_dir, f'{i}_{img_id}.tif')), img)
        # exit()


if __name__ == '__main__':
    生成若干张标记坐标的图像给专家评估准确性()
