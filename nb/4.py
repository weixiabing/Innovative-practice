
import pandas as pd
import numpy as np
df = pd.DataFrame([(0.0,  np.nan, -1.0, 1.0),
                    (np.nan, 2.0, np.nan, np.nan),
                   (2.0, 3.0, np.nan, 9.0),
                    (np.nan, 4.0, -4.0, 16.0)],
                   columns=list('abcd'))

df['d']=df['d'].fillna(df['d'].interpolate(method='linear', limit_direction='forward', axis=0))

print(df)