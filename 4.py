import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
pf=pd.read_csv('library/csv/pm/beijing_final1.csv')
pf2=pd.read_csv('library/csv/pm/beijing_final2.csv')

model = ExponentialSmoothing(pf["Value"], trend="add", seasonal="add", seasonal_periods=8760).fit()
pred = model.forecast(8784)
pred.to_csv("library/csv/pm/beijing_result.csv", encoding="utf_8_sig")
plt.figure()
plt.plot( range(8784),pf2['Value'], color='r', label='Value')
plt.plot(range(8784), pred , color='b', label='pred')

plt.legend()
plt.show()
