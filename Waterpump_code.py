# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 11:06:01 2019

@author: Administrator
"""

import pandas as pd 
import numpy as np
##### Data Cleaning ##### 
#import dataset 
training_feature = pd.read_csv('Training_Feature.csv')
training_label=pd.read_csv('Training_Label.csv')
training=pd.merge(training_feature, training_label, how='inner', left_on='id', right_on='id')
training.info()
training.dtypes
print(training.describe(include=['O']))
# Check the null values 
training.isnull().any()
(len(training)-training.count())/len(training)
# Now we see that all the null values come from categorical values, we chose to fill those null values with category'unknown'
training=training.fillna('unknown')
training.to_csv(r"G:\Duke MQM\Data Competition\Water Pump\training.csv")

##### Model Building and Optimization ##### 
## Modelling Preparation 
# remove unnecessary variables: date_recorded, id, wpt_name, recorded_by
# also removed subvillage, funder and ward to avoid high dimensionality and increase running speed 
training2=training.drop(['date_recorded','id','wpt_name','recorded_by','status_group','subvillage','funder'], axis=1)
# convert code to category 
#training2['region_code']=training2['region_code'].astype("category")
#training2['district_code']=training2['district_code'].astype("category")

# Use Minmax scaler to normalize data
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
training2_num=training2.loc[:, ['amount_tsh','gps_height','longitude','latitude','num_private','population','construction_year']]
scaler.fit(training2_num)
training2_num_scaled=pd.DataFrame(scaler.transform(training2_num))
training2_num_scaled.columns=['amount_tsh','gps_height','longitude','latitude','num_private','population','construction_year']

# create dummy variable for categories 
training2_cat=training2.drop(['amount_tsh','gps_height','longitude','latitude','num_private','population','construction_year'], axis=1)
training2_cat.nunique()
training2_dummy=pd.get_dummies(training2_cat)

training3=pd.merge(training2_num_scaled, training2_dummy, how='inner', left_index=True, right_index=True)


X=training3
# convert y from categorical values to numerical codes for future convenience 
y=training['status_group'].astype('category').cat.codes
# 2: non-functional, 1: functional needs repair, 0: functional 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0)

## Decision Tree 
from sklearn.tree import DecisionTreeClassifier
Decision=DecisionTreeClassifier().fit(X_train, y_train)

# feature importance 
feature_importance=Decision.feature_importances_
features=pd.DataFrame(X_train.columns, feature_importance).reset_index()
features.columns=['importance','features']
# We can see there are many features that have no importance at all, so we want to exclude those featueres 
features_selected=features[features['importance']>0]['features']
# Now we reduced features from 5151 to 1120 

X_train2=X_train.loc[:][features_selected]
X_test2=X_test.loc[:][features_selected]

Decision=DecisionTreeClassifier().fit(X_train2, y_train)
print("Accuracy of Decision Tree on training set: {:.2f}".format(Decision.score(X_train2, y_train)))
print("Accuracy of Decision Tree on test set: {:.2f}".format(Decision.score(X_test2, y_test)))

## Logistic Regression 
# use for loop 
from sklearn.linear_model import LogisticRegression 
C_val = [1,3,5,7]
train_acc=[0.744,0.743,0.747,0.743]
test_acc=[0.738,0.737,0.741,0.737]
for n in range(4):
    print(C_val[n], 'chosen')
    logistic2=LogisticRegression(solver='lbfgs', C=C_val[n], multi_class='auto').fit(X_train2, y_train)
    print('training accuracy:', logistic2.score(X_train2, y_train))
    print('testing accuracy:', logistic2.score(X_test2, y_test))
# currently C=5 is the best choice among the 5 options, and the best test accuracy is 74.1%

# visualize the tuning 
import matplotlib.pyplot as plt
plt.figure()    
plt.plot(C_val, train_acc, '-',C_val, test_acc,'-')
plt.title('training and testing accuracy for different C values')
plt.ylim((0.7, 0.8))
plt.xticks([1,3,5,7])
plt.xlabel('C values')
plt.ylabel('Prediction Accuracy')
plt.text(2, 0.75, 'training accuracy')
plt.text(2, 0.73, 'testing accuracy')
plt.vlines(x=5, ymin=0.7, ymax=0.8, linestyles='dotted')
plt.show()

## Random Forest 
from sklearn.ensemble import RandomForestClassifier 
Rf=RandomForestClassifier(max_features=20, random_state=0, n_estimators=100, class_weight='balanced_subsample')
Rf.fit(X_train2, y_train)
print("Accuracy of Random Forest on training set: {:.2f}".format(Rf.score(X_train2, y_train)))
print("Accuracy of Random Forest on testing set: {:.2f}".format(Rf.score(X_test2, y_test)))
# No Scaler, 8 features: It has training accuracy of 0.98 and testing accuracy of 0.79 
# With MinMaxScaler, 8 features: No, change, still 0.98 and 0.79 
# Optimize the Random Forest by changing the max_features 
# 20 features: [0.98, 0.8]
# 50 features: [0.98,0.79]
# sqrt features: [0.98,0.79] 
# 70 features: [0.98, 0.8]
# reduce dimensionality using feature importance doesn't affect the accuracy, still [0.98,0.8] with 70 features and 100 estimators 

## Gradient Boosting Classifier 
from sklearn.ensemble import GradientBoostingClassifier 
Gradient=GradientBoostingClassifier(random_state=0, learning_rate=0.1, max_depth=10, n_estimators=100) 
Gradient.fit(X_train2, y_train)
print("Accuracy of Gradient Boosted Decision Tree on training set: {:.2f}".format(Gradient.score(X_train2, y_train)))
print("Accuracy of Gradient Boosted Decision Tree on testing set: {:.2f}".format(Gradient.score(X_test2, y_test)))
# it has training accuracy of 0.76 and testing accuracy of 0.76 (learning_rate=0.1, n_estimators=100, max_depth=3)
# it has training accuracy of 0.78 and testing accuracy of 0.77 (learning_rate=0.1, n_estimators=100, max_depth=4)
# it has training accuracy of 0.81 and testing accuracy of 0.78 (learning_rate=0.1, n_estimators=100, max_depth=6)
# it has training accuracy of 0.88 and testing accuracy of 0.80 (learning_rate=0.1, n_estimators=100, max_depth=10)
# it has training accuracy of 0.96 and testing accuracy of 0.80 (learning_rate=0.1, n_estimators=100, max_depth=15)

## Deep Learning 
from sklearn.neural_network import MLPClassifier 
mlp=MLPClassifier(hidden_layer_sizes=[100,100,100], solver='lbfgs', random_state=0).fit(X_train2, y_train)
print("Accuracy of Neural Network on training set: {:.2f}".format(mlp.score(X_train2, y_train)))
print("Accuracy of Neural Network on test set: {:.2f}".format(mlp.score(X_test2, y_test)))
# Current training and testing accuracy 0.75


## Model Selection 
# Since both RF and GradientBosstingClassifier could achieve 80% accuracy, and RF runs slightly faster
# We decided to choose RF model and predict the results 

y_test_prediction=pd.Series(Rf.predict(X_test2), index=X_test2.index)
predict_prob=pd.DataFrame(Rf.predict_proba(X_test2), index=X_test2.index)
predict=training.iloc[X_test2.index]
predict['y_predict']=y_test_prediction
predict['y_actual']=predict['status_group'].astype('category').cat.codes
predict2=pd.merge(predict, predict_prob, how='inner', left_index=True, right_index=True)
predict2.to_csv(r"G:\Duke MQM\Data Competition\Water Pump\prediction.csv")