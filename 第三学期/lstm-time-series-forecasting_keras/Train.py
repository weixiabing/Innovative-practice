import tensorflow as tf
import numpy as np
import os
import h5py
from  Config import  Config
from  lstm.LSTM_Interface import  start_Train
import pandas as pd

config = Config()

if not os.path.exists(config.path): os.makedirs(config.path)



f = h5py.File('D:\library\Github\Innovative-practice\library\csv\第三学期\data_git_version.h5', 'r')
#注:为了演示方便故不使用wnd_dir，其实可以通过代码将其转换为数字序列
#将f['data']转化成numpy数组
data = np.array(f['data'])


model,normalize = start_Train(data,config)

model.save(config.path+config.dimname+".h5")
np.save(config.path+config.dimname+".npy",normalize)
