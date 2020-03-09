import pandas as pd
import numpy as np
import os
os.chdir('D:\\PYTHON\\stayalert')
#import train and test
train =pd.read_csv('fordTrain.csv')
val=pd.read_csv('fordTest.csv')
#audit
train.head()
train.tail()
train.dtypes
summary=train.describe()
skewness=train.skew()
#drop variable p8,v7 and v9
train_1=train.drop(['P8','V7','V9'],axis=1)
#missing values
train_1.isna().sum()
#correlation matrix
cormat=train_1.corr()
cormat.to_csv('cor.csv')
train_1.head()
y=train_1['IsAlert']
x=train_1.iloc[:,3:30]
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=123)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)

preds_lr = lr.predict(x_train)
lr.coef_
lr.intercept_
from sklearn.metrics import confusion_matrix
cm_lr = confusion_matrix(y_train, preds_lr)

preds_val_lr=lr.predict(x_test)
cm_val_lr=confusion_matrix(y_test,preds_val_lr)

from sklearn.preprocessing import normalize
skew_scale=x_train.skew()
x_train_scale=normalize(x_train)
lr.fit(x_train_scale,y_train)
preds_lr_normal=lr.predict(x_train_scale)
cm_lr_scale=confusion_matrix(y_train,preds_lr_normal)
x_train_scale=pd.DataFrame(x_train_scale)
skewness_scale=x_train_scale.skew()

### Decision Tree
from sklearn.tree import DecisionTreeClassifier
DTC= DecisionTreeClassifier()
DTC.fit(x_train,y_train)
pred_dt=DTC.predict(x_train)
cm_dt=confusion_matrix(y_train,pred_dt)
pred_dt_test=DTC.predict(x_test)
cm_dt_test=confusion_matrix(y_test,pred_dt_test)

### Validation data

#summary=val.describe()
x_val=val.drop(['P8','V7','V9'],axis=1)
x_val=x_val.iloc[:,3:30]
preds_dtc_val=DTC.predict(x_val)
solution=pd.read_csv('Solution.csv')
cm_dtc_final=confusion_matrix(solution['Prediction'],preds_dtc_val)

