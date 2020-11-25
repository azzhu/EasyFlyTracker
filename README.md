
<div align='center'>

![logo](imgs/logo.jpg)
</div>


**An easy-to-use program for analyzing Drosophila Activity.**

<div align='center'>

![gif](imgs/gif.gif)
</div>

---

## Features

* 快速的果蝇跟踪

* 支持同时追踪多个腔室的果蝇

* 可自由选择特定的腔室

* 支持分组对比

* 可以选取视频中感兴趣的时间段来分析

## Installation

#### Online installation

Install the [PyPI package](https://pypi.org/project/easyFlyTracker/):

```commandline
pip install easyFlyTracker
```

or

```commandline
pip install -i https://pypi.org/simple/ easyFlyTracker
```

#### Or local installation

Clone the repository:

```commandline
git clone https://github.com/azzhu/EasyFlyTracker.git
```

or [download and extract the zip](https://github.com/azzhu/EasyFlyTracker/archive/master.zip) into your project folder.

Then install it using the local installation command:

```commandline
python setup.py build
python setup.py install
```

## Usage

程序包含两个命令：

#### easyFlyTracker

* SYNOPSIS
```commandline
easyFlyTracker [config_file_path]
easyFlyTracker -h 
easyFlyTracker --help
```

* DESCRIPTION

该命令主要用来跟踪果蝇，并保存跟踪的结果。
接收一个命令行参数，当传递配置文件路径时，程序正常运行。有关配置文件中参数详细说明参见[config.yaml](https://github.com/azzhu/EasyFlyTracker/blob/master/config.yaml).
如果想查看该命令的帮助信息，可以传递 ***-h*** 或者 ***--help*** 来查看。

#### easyFlyTracker_analysis

* SYNOPSIS
```commandline
easyFlyTracker_analysis [config_file_path]
easyFlyTracker_analysis -h 
easyFlyTracker_analysis --help
```

* DESCRIPTION

该命令主要用于分析跟踪的结果，并以图形的方式展示。
接收一个命令行参数，当传递配置文件路径时，程序正常运行。有关配置文件中参数详细说明参见[config.yaml](https://github.com/azzhu/EasyFlyTracker/blob/master/config.yaml).
如果想查看该命令的帮助信息，可以传递 ***-h*** 或者 ***--help*** 来查看。

## Quick Start

[tests](https://github.com/azzhu/EasyFlyTracker/tree/master/tests) 文件夹下有demo视频，您可以拿该数据来快速上手。

1. 在config.yaml中设置正确的视频路径、输出文件夹路径以及其他关注的参数；

2. 跟踪果蝇，运行命令：
    ```commandline
    easyFlyTracker [your config file path]
    ```
3. 分析跟踪结果，运行命令：
    ```commandline
    easyFlyTracker_analysis [config_file_path]
    ```
    等待分析完成，结果保存在输出文件夹中。

## License

EasyFlyTracker is released under the [MIT license](https://github.com/azzhu/EasyFlyTracker/blob/master/LICENSE).