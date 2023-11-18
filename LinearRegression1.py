from bioinfokit.analys import stat, get_data
import numpy as np
import pandas as pd

df = get_data('CancerData.csv').data
print(df)

X = df['']
y = df['Classification']

from sklearn.linear_model import LinearRegression

X = np.array(X).reshape(-1, 1)
y = np.array(y)
reg = LinearRegression().fit(X, y)

# find the slope of best fit line
print(reg.coef_)

# find y intercept
print(reg.intercept_)

'''
predict target variable (yhat) at a given X value
reg.predict([[X]])
'''

import statsmodels.api as sm

# add intercept
X = sm.add_constant(X)

# fit the model
reg = sm.OLS(y, X).fit()
print(reg.summary())
