
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.DataFrame()
df=pd.read_csv('E:/library/Github/Innovative-practice/library/csv/pm/beijing.csv')
print(df)
df.plot()
plt.show()