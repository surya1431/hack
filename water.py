# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 10:21:15 2019

@author: Surya
"""

import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
df = pd.read_csv('F:\python 372\Py_ex\data.csv')
x=df['quantity']
y1=df['COD']
df = pd.DataFrame(x)
#df.columns = df.feature_names
#analyzing the data with some visualization using seaborn library.
sns.Countplot(x = "zn" ,df) #gives the histogram with the values can do similar to others by just changing the label
df.info()#give you the full detailed info about the data that you are going to use.
train_x, test_x, train_y1, test_y1 = model_selection.train_test_split(x, y1, testsize = 0.3, random_state = 1) # we are splitting the data training(70%) and test into 30 for precise result
train_x = train_x.values.reshape(-1,1)
test_x = test_x.values.reshape(-1,1)
alg = LinearRegression()
alg.fit(train_x,train_y1)
y1_pred = alg.predict(test_x)
x_inp=int(input("Enter Quantity water: "))
y1_out=alg.predict([[x_inp]])
plt.scatter(y1_pred,test_x)
plt.show()
print('R2 Score:',r2_score(test_y1,y1_pred))
if(y1_out>=150):
    print("Try with Fenton reaction method")
else:
    print("Quantity of water in chemicals is normal")
