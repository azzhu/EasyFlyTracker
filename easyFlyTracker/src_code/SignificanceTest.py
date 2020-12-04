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

    def __init__(self, dirs):
        '''
        把Qudashen的儿子给初始化
        :param dirs: 文件夹路径的list，根据list的长度来判断是单个文件夹还是多个文件夹
        '''
        # 调用父类的初始化，也就是说把曲大妈给初始化。
        super().__init__()
        self.dirs = dirs
        self.__init_outputdirs()
        print()

    def __init_outputdirs(self):
        '''
        初始化要输出的各种文件的文件名list
        :return:
        '''
        stem = 'mingzi'  # 要保存的文件的名字，【这个地方需要修改】
        self.output_dirs = []
        for d in self.dirs:
            dic = {}
            dic['img_path'] = str(Path(d, 'plot_images', f'{stem}.png'))
            dic['npy_path'] = str(Path(d, 'plot_images/.npy', f'{stem}.npy'))
            dic['excel_path'] = str(Path(d, f'{stem}.xlsx'))
            self.output_dirs.append(dic)


if __name__ == '__main__':
    # 测试代码在这里写 ##############

    # param
    dirs = [
        r'D:\Pycharm_Projects\qu_holmes_su_release\tests\output2',
        r'D:\Pycharm_Projects\qu_holmes_su_release\tests\output3',
    ]
    # 实例化一个对象s，即Sign类的一个对象，或者说Qudashen的儿子的对象，或者说Qudashen的儿媳妇
    s = Sign(dirs=dirs)

    print(dir(s))
