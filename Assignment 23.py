#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import statsmodels.api as sm

file_path = "Civil_Engineering_Regression_Dataset.csv"
df = pd.read_csv(file_path)

X = df[['Building_Height', 'Material_Quality_Index', 'Labor_Cost', 'Concrete_Strength', 'Foundation_Depth']]
y = df['Construction_Cost']

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

regression_coefficients = model.params

highest_impact_variable = regression_coefficients[1:].abs().idxmax()

print("Regression Coefficients:")
print(regression_coefficients)
print("\nVariable with Highest Impact:", highest_impact_variable)


# In[ ]:




