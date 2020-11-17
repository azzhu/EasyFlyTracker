#!/GPFS/zhangli_lab_permanent/zhuqingjie/env/py3_tf2/bin/python
'''
@Time    : 20/11/16 下午 06:45
@Author  : zhuqingjie 
@User    : zhu
@FileName: distit.py
@Software: PyCharm
'''
import os
import time
import shutil

# delete cache files
shutil.rmtree('dist', ignore_errors=True)
shutil.rmtree('build', ignore_errors=True)
shutil.rmtree('easyFlyTracker.egg-info', ignore_errors=True)

time.sleep(0.5)
# 1，Generating distribution archives
os.system('python setup.py sdist bdist_wheel')

time.sleep(0.5)
# 2，Uploading the distribution archives
os.system('twine upload --repository pypi dist/*')
