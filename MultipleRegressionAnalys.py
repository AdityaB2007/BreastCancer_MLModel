'''
Multicollinearity refers to high correlation in more than two independent variables in a regression model.
It often arises from poorly designed experiments or from creating new independent variables not distinct to the existing ones.
The variance inflation factor (VIF) measures the degree of multicollinearity in the regression model.
For example,
    when VIF = 1 -> complete absence of multicollinearity
    when VIF is in range (1, 2) -> absence of strong multicollinearity
    when VIF > 2 -> presence of moderate to strong multicollinearity
Note that VIF can only detect multicollinearity, but it cannot identify the variables causing it. In this case, regression analysis is highly useful.
The efficiency of a regression model largely depends on the correlation structure of independent variables. As such, multicollinearity leads to inaccurate results.
If it is present in the regression model, multicollinearity leads to a biased and unstable estimation of regression coefficients, increases the variance and standard deviation of the model, and decreases the statistical power.
Multicollinearity can be fixed or managed in multiple ways:
    1. Increasing the sample size
    2. Removing highly correlated independent variables. They can be identified through correlation analysis, and this process does not lead to a loss of information.
    3. Combine the highly correlated independent variables into one.
'''

# Calculate the VIF using CancerData.csv clinical features
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
df = pd.read_csv('CancerData.csv')
X = df[['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 'Leptin', 'Adiponectin', 'Resistin', 'MCP.1']] # independent variables
y = df[['Classification']] # dependent variable
X = sm.add_constant(X)

# fit the regression model
reg = sm.OLS(y, X).fit()

# get variance inflation factor (VIF)
pd.DataFrame({'variables':X.columns[1:], 'VIF':[variance_inflation_factor(X.values, i+1) for i in range(len(X.columns[1:]))]})
