import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
pf=pd.read_csv('library/csv/pm/shanghai_final1.csv')
#pf["1exp"] = SimpleExpSmoothing(pf["Value"]).fit(optimized= True).fittedvalues
pf["3exp_add"] = ExponentialSmoothing(pf["Value"][:20000], trend="add", seasonal="add", seasonal_periods=288).fit().fittedvalues
pf["3exp_mul"] = ExponentialSmoothing(pf["Value"][:20000], trend="mul", seasonal="mul", seasonal_periods=288).fit().fittedvalues
plt.figure()
plt.plot( range(20000),pf['Value'][:20000], color='r', label='Value')
plt.plot(range(20000), pf["3exp_add"] , color='b', label='pred_add')
plt.plot(range(20000), pf["3exp_mul"] , color='g', label='pred_mul')
plt.legend()
plt.show()
