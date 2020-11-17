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
发布自己的Python库官方教程：
https://packaging.python.org/tutorials/packaging-projects/
知乎链接：
https://zhuanlan.zhihu.com/p/66603015?utm_source=qq

1，Generating distribution archives
python setup.py sdist bdist_wheel

2，Uploading the distribution archives
twine upload --repository pypi dist/*

pip list所显示的包名跟import的包名的区别：
pip list显示的名字跟setup函数里的name指定的名字对应；
import的包名跟setup函数里的packages指定的模块名对应（注意，这里可以有多个模块）；

国内源找不到该包，一定要使用官方源安装：
pip install -i https://pypi.org/simple/ easyFlyTracker
'''

with open('requirements.txt') as f:
    req = [line.strip() for line in f.readlines() if line.strip()]

with open("README.md", "r") as fh:
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
    project_urls={
        "Source Code": "https://github.com/azzhu/EasyFlyTracker",
        "Bug Tracker": "https://github.com/azzhu/EasyFlyTracker/issues",
        # "Documentation": "",  # 待补充修改
    },
    entry_points={
        'console_scripts': [
            'easyFlyTracker=easyFlyTracker.cli:_easyFlyTracker',
            # 这里可以继续添加新的命令
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3',
)
