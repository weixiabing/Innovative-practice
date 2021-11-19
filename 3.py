import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
pf=pd.read_csv('library/csv/pm/beijing_final1.csv')
decompose_result = seasonal_decompose(pf, model="multiplicative", period=8760)
decompose_result.plot()
plt.show()