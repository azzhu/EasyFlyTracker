#!/GPFS/zhangli_lab_permanent/zhuqingjie/env/py3_tf2/bin/python
'''
@Time    : 20/11/05 下午 01:56
@Author  : zhuqingjie 
@User    : zhu
@FileName: setup.py
@Software: PyCharm
'''
import setuptools
import easyFlyTracker

''' 
国内源找不到该包(或者说更新不及时)，一定要使用官方源安装：
pip install -i https://pypi.org/simple/ easyFlyTracker
pip install --upgrade -i https://pypi.org/simple/ easyFlyTracker
🀙🀚🀛🀜🀝🀞🀟🀠🀡🀢🀣
🀥🀗🀐🀏🀎🀍🀌🀋🀊🀉
🀙🀚🀛🀜🀝🀞🀟🀠🀡🀢🀣
🀥🀗🀐🀏🀎🀍🀌🀋🀊🀉

'''

with open('requirements.txt') as f:
    req = [line.strip() for line in f.readlines() if line.strip()]

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="easyFlyTracker",
    version=easyFlyTracker.__version__,
    author="azzhu",
    author_email="zhu.qingjie@qq.com",
    description="An easy-to-use program for analyzing Drosophila Activity.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/azzhu/EasyFlyTracker",
    install_requires=req,
    license='MIT',
    packages=setuptools.find_packages(),
    package_data={
        'easyFlyTracker': ['fonts/*.ttf'],
    },
    project_urls={
        "Source Code": "https://github.com/azzhu/EasyFlyTracker",
        "Bug Tracker": "https://github.com/azzhu/EasyFlyTracker/issues",
        # "Documentation": "",  # 待补充修改
    },
    entry_points={
        'console_scripts': [
            'easyFlyTracker=easyFlyTracker.cli:easyFlyTracker_',
            'easyFlyTracker_analysis=easyFlyTracker.cli:easyFlyTracker_analysis',
            'easyFlyTracker_cam_calibration=easyFlyTracker.cli:easyFlyTracker_cam_calibration',
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3',
)
