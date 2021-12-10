#! python
# @Time    : 21/12/09 下午 04:06
# @Author  : azzhu 
# @FileName: kernel_density_estimation.py
# @Software: PyCharm
import numpy as np

from sklearn.neighbors import KernelDensity
import cv2

'''
核密度估计（Kernel Density Estimation，KDE）：
    说白了就是对离散的连续数值进行平滑，常用于直方图，可把直方图估计为一个连续平滑的曲线。

    密度估计，即估计概率密度函数，前面加个核，即使用核的方法来实现。
'''


def get_KernelDensity(data, bins=500, range=None, kernel='gaussian'):
    '''
    核密度估计
    :param data: 数据，统计直方图之前的数据，不是直方图数据
    :param bins: 设置的bing个数
    :param kernel: 核，['gaussian'|'tophat'|'epanechnikov'|'exponential'|'linear'|'cosine']
    :return:
    '''
    data = data.flatten()
    bandwidth = 1.05 * np.std(data) * (len(data) ** (-1 / 5))  # 带宽太小太大都不好，取这个值最优
    model = KernelDensity(bandwidth=bandwidth, kernel=kernel)
    model.fit(data[:, np.newaxis])
    if range is None:
        x_range = np.linspace(data.min() - 1, data.max() + 1, bins)
    else:
        x_range = np.linspace(range[0], range[1], bins)
    x_log_prob = model.score_samples(x_range[:, np.newaxis])  # 这个东西返回概率的对数
    x_prob = np.exp(x_log_prob)
    return x_range, x_prob


if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt

    changes_es = np.load(r'D:\tempp\changes_es.npy', allow_pickle=True)
    hists = []
    histsKDE = []
    for cha in changes_es:
        hist = np.histogram(cha.flatten(), bins=18, range=(0, 180))
        hists.append(hist)
        histKDE = get_KernelDensity(cha, range=(0, 180))
        histsKDE.append(histKDE)
    hists = np.array([h[0] for h in hists])
    xsKDE = histsKDE[0][0]
    histsKDE = np.array([h[1] for h in histsKDE])
    xs = list(range(len(hists[0])))
    # np.save(self.angle_changes_path, hists)
    plt.rcParams['figure.figsize'] = (15.0, 8.0)
    plt.grid(linewidth=1)
    plt.xlabel('Angle region (degree)')
    plt.ylabel('Frequency (times)')
    plt.title('Histogram of angle change per duration')
    for i, (hi, hiKDE) in enumerate(zip(hists, histsKDE)):
        # plt.plot(xs, hi, label=f'Duration {i + 1}')
        plt.plot(xsKDE, hiKDE, label=f'Duration {i + 1}')
        print(f'sumkde:{np.sum(hiKDE)}')  # 和不是一
        # break
    plt.legend(prop={'family': 'Times New Roman', 'size': 12})
    plt.legend(loc='upper right')
    plt.show()

    exit()

    # 正态分布
    x = np.random.normal(10, 1, 1000)

    x_range, x_prob = get_KernelDensity(x, range=(0, 15))

    plt.figure(figsize=(10, 10))
    r = plt.hist(
        x=x,
        bins=50,
        density=True,
        histtype='stepfilled',
        color='red',
        alpha=0.5,
        label='直方图',
    )
    plt.fill_between(
        x=x_range,
        y1=x_prob,
        y2=0,
        color='green',
        alpha=0.5,
        label='KDE',
    )
    plt.plot(x_range, x_prob, color='gray')
    plt.show()
