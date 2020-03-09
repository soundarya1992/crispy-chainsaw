import pandas as pd
import numpy as np
import os
# Working directory
os.chdir('D:\\PYTHON\\bigmart')
#Read files:
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submit = pd.read_csv("Submit.csv")
#Combine both train and test
train['source']='train'
test['source']='test'
data = pd.concat([train, test],ignore_index=True)
print(train.shape, test.shape, data.shape)
#Missing values
data.apply(lambda x: sum(x.isnull()))
#EDA
 #numerical variables
data.describe()
 #catergorical
data.apply(lambda x: len(x.unique()))

data.dtypes
#Filter categorical variables
categorical_columns = [x for x in data.dtypes.index if data.dtypes[x]=='object']
categorical_columns = [x for x in categorical_columns 
                       if x not in ['Item_Identifier','Outlet_Identifier','source']]
#Print frequency of categories
for col in categorical_columns:
    print(data[col].value_counts())

# Data Cleaning
    #Missing values – Item_Weight and Outlet_Size
missing_Itemwt_bf= sum(data['Item_Weight'].isnull())
data['Item_Weight'] = np.where(data['Item_Weight'].isnull(), np.mean(data['Item_Weight']), data['Item_Weight'])
missing_Itemwt_af = sum(data['Item_Weight'].isnull())

missing_outletsize_bf = sum(data['Item_Outlet_Sales'].isnull())
data['Item_Outlet_Sales'] = np.where(data['Item_Outlet_Sales'].isnull(), "Medium", data['Item_Outlet_Sales'])
missing_outletsize_af = sum(data['Item_Outlet_Sales'].isnull())

# Solving data inconsistency
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF':'Low Fat',
                                                             'reg':'Regular',
                                                             'low fat':'Low Fat'})


# Creating new variable total years from Outlet_Establishment_Year
data['total_years']= 2013 - data['Outlet_Establishment_Year']

#Label Encoding
from sklearn.preprocessing import LabelEncoder
#New variable for outlet
le = LabelEncoder()
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Outlet']
data = data.apply(lambda col: le.fit_transform(col.astype(str)), axis=0, result_type='expand')

for i in var_mod:
    data[i] = le.fit_transform(data[i])
    
data.dtypes
# Exporting the data
#Drop the columns which have been converted to different types:
data.drop(['Outlet_Establishment_Year'],axis=1,inplace=True)
#Divide into test and train:
train = data.loc[data['source']==1]
test = data.loc[data['source']==0]
#Splitting into labels and features
#Drop unnecessary columns:

y = train.Item_Outlet_Sales
x=train.drop('source',axis=1)
x=x.drop('Item_Outlet_Sales', axis=1)

test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)

#Split
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=123)

#Export files as modified versions:
train.to_csv("train_modified.csv",index=False)
test.to_csv("test_modified.csv",index=False)

#Regression Model
# Model initialization
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
regression_model = LinearRegression()
# Fit the data(train the model)
regression_model.fit(x_train, y_train)
regression_model.coef_
regression_model.intercept_
regression_model.r_sq
regression_model._residues
# Predict
y_train_predicted = regression_model.predict(x_train)
# model evaluation
rmse_train = sqrt(mean_squared_error(y_train, y_train_predicted))
#Predict for test
y_test_predicted = regression_model.predict(x_test)
#evaluation
rmse_test = sqrt(mean_squared_error(y_test, y_test_predicted))
#Predict for test
test_pred = regression_model.predict(test)
#Submission
test_pred=pd.DataFrame(test_pred)
Item_Outlet_Sales = test_pred
submit=pd.concat([submit, Item_Outlet_Sales], axis=1, ignore_index=True)
submit.to_csv("Submit.csv",index=False)

import seaborn as sns
import matplotlib.pyplot as plt
sns.residplot(y_train, y_train_predicted, lowess=True, color="g")
sns.residplot(y_test, y_test_predicted, lowess=True, color="g")
plt.show()
#SVM

from sklearn.svm import LinearSVR
clf = LinearSVR() 
clf.fit(x_train, y_train) 
y_train_predsvr = clf.predict(x_train)
rmse_trainsvr = sqrt(mean_squared_error(y_train, y_train_predsvr))
y_test_predsvr=clf.predict(x_test)
rmse_testsvr = sqrt(mean_squared_error(y_test, y_test_predsvr))
y_test_svr = clf.predict(test)
y_test_svr=pd.DataFrame(y_test_svr)
Item_Outlet_Sales=y_test_svr
submit_1=pd.concat([submit, Item_Outlet_Sales], axis=1)
submit_1.to_csv("Submit.csv",index=False, header=True)






