# -*- coding: utf-8 -*-
# 导入包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False

# 导入数据
df = pd.read_csv('D:\library\Github\Innovative-practice\library\csv\第三学期/testlist.csv')
df_coor = df.corr()
print(df_coor)

fig, ax = plt.subplots(figsize=(6, 6),facecolor='w')
# 指定颜色带的色系
sns.heatmap(df.corr(),annot=True, vmax=1, square=True, cmap="Blues", fmt='.2g')
plt.title('相关性热力图')
plt.show()
fig.savefig('D:\library\Github\Innovative-practice\第三学期\DATA\pic\df_corr.png',bbox_inches='tight',transparent=True)
