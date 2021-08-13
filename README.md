
<div align='center'>

![logo](imgs/logo.jpg)
</div>

<div align='center'>

💜 [easyFlyTracker.cibr.ac.cn](http://easyFlyTracker.cibr.ac.cn/) 💜
</div>

***EasyFlyTracker is an easy-to-use Python 3-based package that can track and analyze Drosophila sleep and locomotor activity based on video shooting. It can be used for high-throughput simultaneous tracking and analysis of drug-treated individual adult fly. This software will accelerate basic research on drug effect studies with fruit flies.***

<div align='center'>

![gif](imgs/gif.gif)
</div>

---

## Features

* EasyFlyTracker is open-source.
* EasyFlyTracker is easy to use and fast.
* EasyFlyTracker is easily expandable.
* EasyFlyTracker supports simultaneous tracking of Drosophila with multiple Chambers and low-cost.
* EasyFlyTracker supports the selection of specific flies.
* EasyFlyTracker supports group tracking of Drosophila.
* EasyFlyTracker supports different outputs.
* Drug-treatment study example of EasyFlyTracker is provided.
* All the products lists are provided for your information.

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

or 

```commandline
conda install easyFlyTracker
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

## Documentation

💜 EasyFlyTracker documentation: http://easyflytracker.cibr.ac.cn/#/document

## Forum

💜 EasyFlyTracker forum: http://easyflytracker.cibr.ac.cn/#/suggest

## Usage

The program contains two commands:

#### easyFlyTracker

* SYNOPSIS
```commandline
easyFlyTracker [config_file_path]
easyFlyTracker -h 
easyFlyTracker --help
```

* DESCRIPTION

This command is used to track fruit flies and save the results.
Receiving a command line argument, the program runs normally when passing the configuration file path.For details about the parameters in the configuration file, see [config.yaml](https://github.com/azzhu/EasyFlyTracker/blob/master/tests/config.yaml).
To view the help information about the command, run the command with *-h* or *--help* param.

#### easyFlyTracker_analysis

* SYNOPSIS
```commandline
easyFlyTracker_analysis [config_file_path]
easyFlyTracker_analysis -h 
easyFlyTracker_analysis --help
```

* DESCRIPTION

This command is used to analyze tracing results and display them graphically.
Receiving a command line argument, the program runs normally when passing the configuration file path.For details about the parameters in the configuration file, see [config.yaml](https://github.com/azzhu/EasyFlyTracker/blob/master/tests/config.yaml).
To view the help information about the command, run the command with *-h* or *--help* param.

## Quick Start

There are demo videos in the [tests](https://github.com/azzhu/EasyFlyTracker/tree/master/tests) folder. You can use the data to get started quickly.

1. Set the correct video path, output folder path, and other parameters of interest in config.yaml;

2. To track the flies, run the command:
    ```commandline
    easyFlyTracker [your config file path]
    ```
3. To analyze the trace results, run the command:
    ```commandline
    easyFlyTracker_analysis [config_file_path]
    ```
    Wait for the analysis to complete and the results are saved in the output folder.
   

## Useful Links

💜 EasyFlyTracker homepage: http://easyFlyTracker.cibr.ac.cn/

💜 CIBR homepage: http://www.cibr.ac.cn/

## License

EasyFlyTracker is released under the [MIT license](https://github.com/azzhu/EasyFlyTracker/blob/master/LICENSE).
