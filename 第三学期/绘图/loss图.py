import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
from keras.models import load_model
import os
#解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False




def smooth_xy(lx, ly):
    """数据平滑处理

    :param lx: x轴数据，数组
    :param ly: y轴数据，数组
    :return: 平滑后的x、y轴数据，数组 [slx, sly]
    """
    x = np.array(lx)
    y = np.array(ly)
    x_smooth = np.linspace(x.min(), x.max(), 300)
    y_smooth = make_interp_spline(x, y)(x_smooth)
    return x_smooth, y_smooth

if __name__ == '__main__':
    y_raw=[0.0494,0.0429,0.0426,0.0423,0.0421,0.042,0.042,
    0.0418,0.0417,0.0417,0.0417,0.0416,0.0416,0.0416,0.0416,
    0.0416,0.0416,0.0416,0.0416,0.0415]

    x_raw = np.arange(1, 21, 1)
    #读取D:\library\Github\Innovative-practice\第三学期\DATA\lstm\Test_call.h5里的loss
    #    

    xy_s = smooth_xy(x_raw, y_raw)
    
    
    # 处理后的平滑曲线
    plt.plot(xy_s[0], xy_s[1])
    plt.title('迭代误差变化图')
    #右上角x轴标签
    plt.xlabel('X:迭代训练次数')
    #右上角y轴标签
    plt.ylabel('Y:训练误差精度')
    #设置x轴刻度
    plt.xticks(np.arange(1, 21, 1))
    plt.savefig('D:\library\Github\Innovative-practice\第三学期\DATA\pic\convlstm\loss.png')
    plt.show()
