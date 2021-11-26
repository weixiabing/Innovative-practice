import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
pf=pd.read_csv('library/csv/pm/shanghai_final1.csv')
pf["1exp"] = SimpleExpSmoothing(pf["Value"]).fit(optimized= True).fittedvalues
plt.figure()
plt.plot( range(35063),pf['Value'], color='r', label='Value')
plt.plot(range(35063), pf["1exp"] , color='b', label='pred')
plt.legend()
plt.show()
