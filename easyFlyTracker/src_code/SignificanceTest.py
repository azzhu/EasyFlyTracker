#! python
# @Time    : 20/12/04 上午 10:17
# @Author  : Qudashen
# @FileName: SignificanceTest.py
# @Software: PyCharm

import numpy as np
from pathlib import Path


class Qudashen(object):
    __doc__ = \
        '''
        万物基于曲大神。
        '''


class Sign(Qudashen):
    __doc__ = 'Qudashen的儿子类，喊Qudashen喊妈妈'
    '''
    不管输入为多少个文件夹，只需要输出一批结果（一个图像、一个npy、一个excel等等）。
    当输入为一个文件夹时，在输入文件夹下输出；
    当输入为多个文件夹时，需要指定输出文件夹。
    '''

    def __init__(self, dirs, candidate_outputdir=None):
        """
        把Qudashen的儿子给初始化。
        :param dirs: 文件夹路径的list，根据list的长度来判断是单个文件夹还是多个文件夹；
        :param candidate_outputdir: 当dirs只有一个文件夹时，不使用该值；当有多个文件夹时，使用该值作为输出文件夹；
        """
        # 调用父类的初始化，也就是说把曲大妈给初始化。
        super().__init__()
        self.dirs = dirs
        # 初始化输出文件的路径
        self.__init_output_file_paths(candidate_outputdir)
        # 加载数据
        self.__load_datas()

    def __init_output_file_paths(self, candidate_outputdir):
        '''
        初始化要输出的各种文件的文件名。
        :return:
        '''
        stem = 'ququququququ'  # 要保存的文件的名字，【这个地方需要修改】
        self.output_files = {}
        if len(self.dirs) == 1:  # 如果只有一个文件夹，把结果按照相应的格式输出到该文件夹
            self.output_files['img_path'] = str(Path(self.dirs[0], 'plot_images', f'{stem}.png'))
            self.output_files['npy_path'] = str(Path(self.dirs[0], 'plot_images/.npy', f'{stem}.npy'))
            self.output_files['excel_path'] = str(Path(self.dirs[0], f'{stem}.xlsx'))
        else:  # 如果有多个文件夹，则统一输出到指定的文件夹中
            if not candidate_outputdir:
                raise ValueError('candidate_outputdir')
            Path(candidate_outputdir).mkdir(exist_ok=True)
            self.output_files['img_path'] = str(Path(candidate_outputdir, f'{stem}.png'))
            self.output_files['npy_path'] = str(Path(candidate_outputdir, f'{stem}.npy'))
            self.output_files['excel_path'] = str(Path(candidate_outputdir, f'{stem}.xlsx'))

    def __load_datas(self):
        '''
        加载数据
        :return:
        '''
        # 先寻找所有符合条件的npy文件
        npyfiles = []
        for d in self.dirs:
            npyfiles += list(Path(d).rglob('avg_dist_per_x_min_*.npy'))
        # 去除merge的结果
        npyfiles = [f for f in npyfiles if 'merge' not in str(f)]
        # 先找出所有的组名
        groups = list(set([f.stem.split('_')[-1] for f in npyfiles]))
        # 每个组下放一个list，代表多个npy的数据
        datas = {g: [] for g in groups}
        for f in npyfiles:
            da = np.load(f)
            g = f.stem.split('_')[-1]
            datas[g].append(da)
        self.datas = datas

    def fly(self):
        '''
        开始起飞。
        所有数据保存在self.datas里面，输出路径在self.output_files中。
        :return:
        '''
        # 3,2,1,0... Fly..................

        # 安全着陆。
        print()


if __name__ == '__main__':
    # 测试代码在这里写 ##############

    # param
    dirs = [
        r'D:\Pycharm_Projects\qu_holmes_su_release\tests\output2',
        # r'D:\Pycharm_Projects\qu_holmes_su_release\tests\output3',
    ]
    # 实例化一个对象s，即Sign类的一个对象，或者说Qudashen的儿子的对象，或者说Qudashen的儿媳妇
    s = Sign(
        dirs=dirs,
        candidate_outputdir=r'D:\Pycharm_Projects\qu_holmes_su_release\tests\sign_output'
    )
    s.fly()


    # print(dir(s))
    # print(s.fly.__doc__)
