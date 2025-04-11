# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 21:05:37 2017

@author: nimish.paneri
"""

#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib as plt
import os
import subprocess
import time
from sklearn.externals import joblib
#from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import confusion_matrix
start=time.clock()
####Test , train split not used###
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(df,df_class, train_size = 0.8)
####
# Read files
#features=['AON','SUB_SERVICE_PROVIDER', 'BALANCE', 'HANDSET_TYPE', 'RECHG_CNT_10DAYS_1', 'RECHG_CNT_10DAYS_2', 'RECHG_CNT_10DAYS_3', 'DAYS_RECHG', 'USG_VOC_OG_30DAYS', 'USG_VOC_OG_30_60DAYS', 'USG_VOC_IC_30DAYS', 'USG_VOC_IC_30_60DAYS', 'USG_DATA_30DAYS', 'USG_DATA_30_60DAYS','USG_LOCAL_30DAYS', 'USG_STD_30DAYS', 'USG_INT_30DAYS','USG_VAS_10DAYS_1', 'USG_VAS_10DAYS_2', 'USG_VAS_10DAYS_3','LAST_USG_DAYS2','CC_CALLS_10DAYS', 'CAMP_CNT','DAYS_CAMP','segmentation','PI_Handset_migration','GENDER','AGE','ICR_RECH_CNT_60DAYS','EKYC','PRIMARY_SITE','MNP_FLAG','B_NO_UNIQ_USER_CNT','CAMP_CNT_1','CAMP_CNT_2','CAMP_CNT_3','OG_ON_MOU_11_20oct','OG_OFF_MOU_11_20oct','SR_CNT_TOP1','SR_CNT_TOP2','ARPU_AUG17','ARPU_SEP17','ARPU_OCT17' ]
features=['AON','BALANCE', 'HANDSET_TYPE', 'RECHG_CNT_10DAYS_1', 'RECHG_CNT_10DAYS_2', 'RECHG_CNT_10DAYS_3', 'DAYS_RECHG', 'USG_VOC_OG_30DAYS', 'USG_VOC_OG_30_60DAYS', 'USG_VOC_IC_30DAYS', 'USG_VOC_IC_30_60DAYS', 'USG_DATA_30DAYS', 'USG_DATA_30_60DAYS','USG_LOCAL_30DAYS', 'USG_STD_30DAYS', 'USG_INT_30DAYS','USG_VAS_10DAYS_1', 'USG_VAS_10DAYS_2', 'USG_VAS_10DAYS_3','LAST_USG_DAYS2','CC_CALLS_10DAYS', 'CAMP_CNT','DAYS_CAMP','segmentation','PI_Handset_migration','GENDER','AGE','ICR_RECH_CNT_60DAYS','EKYC','PRIMARY_SITE','MNP_FLAG','B_NO_UNIQ_USER_CNT','CAMP_CNT_1','CAMP_CNT_2','CAMP_CNT_3','OG_ON_MOU_11_20oct','OG_OFF_MOU_11_20oct','SR_CNT_TOP1','SR_CNT_TOP2','ARPU_AUG17','ARPU_SEP17','ARPU_OCT17' ]
#features=['BALANCE', 'HANDSET_TYPE', 'RECHG_CNT_10DAYS_1', 'RECHG_CNT_10DAYS_2', 'RECHG_CNT_10DAYS_3', 'USG_VOC_OG_30DAYS', 'USG_VOC_OG_30_60DAYS', 'USG_VOC_IC_30DAYS', 'USG_VOC_IC_30_60DAYS', 'USG_DATA_30DAYS', 'USG_DATA_30_60DAYS','USG_LOCAL_30DAYS', 'USG_STD_30DAYS', 'USG_INT_30DAYS','USG_VAS_10DAYS_1', 'USG_VAS_10DAYS_2', 'USG_VAS_10DAYS_3','LAST_USG_DAYS2','CC_CALLS_10DAYS', 'CAMP_CNT','DAYS_CAMP','segmentation','PI_Handset_migration',,'GENDER','AGE','ICR_RECH_CNT_60DAYS','EKYC','PRIMARY_SITE','MNP_FLAG','B_NO_UNIQ_USER_CNT','CAMP_CNT_1','CAMP_CNT_2','CAMP_CNT_3','OG_ON_MOU_11_20oct','OG_OFF_MOU_11_20oct','SR_CNT_TOP1','SR_CNT_TOP2','ARPU_AUG17','ARPU_SEP17','ARPU_OCT17' ]

#for vlr analysis only
#features=['B_NO_UNIQ_USER_CNT','USG_STD_30DAYS','USG_INT_30DAYS','LAST_USG_DAYS2' ]
#features= ['USG_VOC_OG_30DAYS','USG_STD_30DAYS','B_NO_UNIQ_USER_CNT']

#With growth feature
#features=['RECHG_GROWTH1','RECHG_GROWTH','USG_VOC_GROWTH','IC_GROWTH','ARPU_GROWTH1','ARPU_GROWTH2','AON','SUB_SERVICE_PROVIDER', 'BALANCE', 'HANDSET_TYPE', 'RECHG_CNT_10DAYS_1', 'RECHG_CNT_10DAYS_2', 'RECHG_CNT_10DAYS_3', 'DAYS_RECHG', 'USG_VOC_OG_30DAYS', 'USG_VOC_OG_30_60DAYS', 'USG_VOC_IC_30DAYS', 'USG_VOC_IC_30_60DAYS', 'USG_DATA_30DAYS', 'USG_DATA_30_60DAYS','USG_LOCAL_30DAYS', 'USG_STD_30DAYS', 'USG_INT_30DAYS','USG_VAS_10DAYS_1', 'USG_VAS_10DAYS_2', 'USG_VAS_10DAYS_3','LAST_USG_DAYS','CC_CALLS_10DAYS', 'CAMP_CNT','DAYS_CAMP','segmentation','PI_Handset_migration','GENDER','AGE','ICR_RECH_CNT_60DAYS','EKYC','PRIMARY_SITE','MNP_FLAG','B_NO_UNIQ_USER_CNT','CAMP_CNT_1','CAMP_CNT_2','CAMP_CNT_3','OG_ON_MOU_11_20oct','OG_OFF_MOU_11_20oct','SR_CNT_TOP1','SR_CNT_TOP2','ARPU_AUG17','ARPU_SEP17','ARPU_OCT17' ]
# Limited Features
#features=['CC_CALLS_10DAYS','USG_VOC_OG_30DAYS', 'USG_VOC_OG_30_60DAYS', 'USG_VOC_IC_30DAYS', 'USG_VOC_IC_30_60DAYS','EKYC','HANDSET_TYPE', 'RECHG_CNT_10DAYS_1', 'RECHG_CNT_10DAYS_2', 'RECHG_CNT_10DAYS_3', 'DAYS_RECHG','MNP_FLAG','OG_OFF_MOU_11_20oct', 'USG_DATA_30DAYS', 'USG_DATA_30_60DAYS']
#features=['AON','HANDSET_TYPE','RECHG_CNT_10DAYS_1', 'RECHG_CNT_10DAYS_2', 'RECHG_CNT_10DAYS_3','DAYS_RECHG','USG_VOC_OG_30DAYS', 'USG_VOC_OG_30_60DAYS', 'USG_VOC_IC_30DAYS', 'USG_VOC_IC_30_60DAYS','CC_CALLS_10DAYS','EKYC','MNP_FLAG','OG_OFF_MOU_11_20oct','USG_DATA_30DAYS', 'USG_DATA_30_60DAYS']
#Experiment
#features=[ 'PRIMARY_SITE' ]
#Shortlisted1
#features=[ 'DAYS_RECHG','CAMP_CNT_1','CAMP_CNT_2','CAMP_CNT_3','LAST_USG_DAYS2','PRIMARY_SITE','SR_CNT_TOP1','SR_CNT_TOP2','ARPU_AUG17','ARPU_SEP17','ARPU_OCT17','PRIMARY_SITE']
#shortlisted2
#features=['AON','BALANCE', 'DAYS_RECHG','USG_VOC_OG_30DAYS','CAMP_CNT_1','CAMP_CNT_2','LAST_USG_DAYS2','PRIMARY_SITE','SR_CNT_TOP1','SR_CNT_TOP2','ARPU_AUG17','ARPU_SEP17','ARPU_OCT17','PRIMARY_SITE']
#shortlisted3
#features=[ 'DAYS_RECHG','CAMP_CNT_1','LAST_USG_DAYS2','PRIMARY_SITE','SR_CNT_TOP1','ARPU_OCT17','PRIMARY_SITE']
#shortlisted4
#features=['BALANCE', 'DAYS_RECHG','USG_VOC_OG_30DAYS','CAMP_CNT_1','CAMP_CNT_2','PRIMARY_SITE','ARPU_AUG17','ARPU_SEP17','ARPU_OCT17','LAST_USG_DAYS2']
#features=['USG_VOC_OG_30DAYS', 'LAST_USG_DAYS2','ARPU_OCT17','ARPU_AUG17','ARPU_SEP17']

#feature with max mean differences btn 0 & 1
#features=['BALANCE','USG_VOC_OG_30DAYS','USG_VOC_OG_30_60DAYS','USG_VOC_IC_30DAYS','USG_VOC_IC_30_60DAYS','USG_DATA_30DAYS','USG_DATA_30_60DAYS','USG_LOCAL_30DAYS','B_NO_UNIQ_USER_CNT','OG_ON_MOU_11_20oct','OG_OFF_MOU_11_20oct','ARPU_AUG17','ARPU_SEP17','ARPU_OCT17']



dfTrain = pd.read_csv("G:\\Churn Project\\data7.csv") # importing train dataset
dfTrain.set_index(['SUB_MSISDN'],inplace=True)
dfTest = pd.read_csv("G:\\Churn Project\\data8_nov.csv") # importing train dataset
dfTest.set_index(['SUB_MSISDN'],inplace=True)
dfTemp= pd.read_csv("G:\\Churn Project\\data8_nov.csv")

# Feature Scalling
#scaleFeatures=['SUB_SERVICE_PROVIDER','AON','BALANCE','RECHG_CNT_10DAYS_1','RECHG_CNT_10DAYS_2','RECHG_CNT_10DAYS_3','DAYS_RECHG','USG_VOC_OG_30DAYS','USG_VOC_OG_30_60DAYS','USG_VOC_IC_30DAYS','USG_VOC_IC_30_60DAYS','USG_DATA_30DAYS','USG_DATA_30_60DAYS','USG_LOCAL_30DAYS','USG_STD_30DAYS','USG_INT_30DAYS','USG_VAS_10DAYS_1','USG_VAS_10DAYS_2','USG_VAS_10DAYS_3','LAST_USG_DAYS2','CC_CALLS_10DAYS','CAMP_CNT','DAYS_CAMP','AGE','ICR_RECH_CNT_60DAYS','USG_FLAG_04_10','CAMP_CNT_1','CAMP_CNT_2','CAMP_CNT_3','OG_ON_MOU_11_20oct','OG_OFF_MOU_11_20oct','SR_CNT_TOP1','SR_CNT_TOP2','ARPU_AUG17','ARPU_SEP17','ARPU_OCT17']
#scaler = MinMaxScaler()
#dfTrain[scaleFeatures]=scaler.fit_transform(dfTrain[scaleFeatures])
#dfTest[scaleFeatures]=scaler.fit_transform(dfTest[scaleFeatures])
#dfTrain[['AON','BALANCE','RECHG_CNT_10DAYS_1','RECHG_CNT_10DAYS_2','RECHG_CNT_10DAYS_3','DAYS_RECHG','USG_VOC_OG_30DAYS','USG_VOC_OG_30_60DAYS','USG_VOC_IC_30DAYS','USG_VOC_IC_30_60DAYS','USG_DATA_30DAYS','USG_DATA_30_60DAYS','USG_LOCAL_30DAYS','USG_STD_30DAYS','USG_INT_30DAYS','USG_VAS_10DAYS_1','USG_VAS_10DAYS_2','USG_VAS_10DAYS_3','LAST_USG_DAYS','CC_CALLS_10DAYS','CAMP_CNT','DAYS_CAMP','AGE','ICR_RECH_CNT_60DAYS','USG_FLAG_04_10','CAMP_CNT_1','CAMP_CNT_2','CAMP_CNT_3','OG_ON_MOU_11_20oct','OG_OFF_MOU_11_20oct','SR_CNT_TOP1','SR_CNT_TOP2','ARPU_AUG17','ARPU_SEP17','ARPU_OCT17']]=scaler.fit_transform(dfTrain[['AON','BALANCE','RECHG_CNT_10DAYS_1','RECHG_CNT_10DAYS_2','RECHG_CNT_10DAYS_3','DAYS_RECHG','USG_VOC_OG_30DAYS','USG_VOC_OG_30_60DAYS','USG_VOC_IC_30DAYS','USG_VOC_IC_30_60DAYS','USG_DATA_30DAYS','USG_DATA_30_60DAYS','USG_LOCAL_30DAYS','USG_STD_30DAYS','USG_INT_30DAYS','USG_VAS_10DAYS_1','USG_VAS_10DAYS_2','USG_VAS_10DAYS_3','LAST_USG_DAYS','CC_CALLS_10DAYS','CAMP_CNT','DAYS_CAMP','AGE','ICR_RECH_CNT_60DAYS','USG_FLAG_04_10','CAMP_CNT_1','CAMP_CNT_2','CAMP_CNT_3','OG_ON_MOU_11_20oct','OG_OFF_MOU_11_20oct','SR_CNT_TOP1','SR_CNT_TOP2','ARPU_AUG17','ARPU_SEP17','ARPU_OCT17']])
#dfTest[['AON','BALANCE','RECHG_CNT_10DAYS_1','RECHG_CNT_10DAYS_2','RECHG_CNT_10DAYS_3','DAYS_RECHG','USG_VOC_OG_30DAYS','USG_VOC_OG_30_60DAYS','USG_VOC_IC_30DAYS','USG_VOC_IC_30_60DAYS','USG_DATA_30DAYS','USG_DATA_30_60DAYS','USG_LOCAL_30DAYS','USG_STD_30DAYS','USG_INT_30DAYS','USG_VAS_10DAYS_1','USG_VAS_10DAYS_2','USG_VAS_10DAYS_3','LAST_USG_DAYS','CC_CALLS_10DAYS','CAMP_CNT','DAYS_CAMP','AGE','ICR_RECH_CNT_60DAYS','USG_FLAG_04_10','CAMP_CNT_1','CAMP_CNT_2','CAMP_CNT_3','OG_ON_MOU_11_20oct','OG_OFF_MOU_11_20oct','SR_CNT_TOP1','SR_CNT_TOP2','ARPU_AUG17','ARPU_SEP17','ARPU_OCT17']]=scaler.fit_transform(dfTest[['AON','BALANCE','RECHG_CNT_10DAYS_1','RECHG_CNT_10DAYS_2','RECHG_CNT_10DAYS_3','DAYS_RECHG','USG_VOC_OG_30DAYS','USG_VOC_OG_30_60DAYS','USG_VOC_IC_30DAYS','USG_VOC_IC_30_60DAYS','USG_DATA_30DAYS','USG_DATA_30_60DAYS','USG_LOCAL_30DAYS','USG_STD_30DAYS','USG_INT_30DAYS','USG_VAS_10DAYS_1','USG_VAS_10DAYS_2','USG_VAS_10DAYS_3','LAST_USG_DAYS','CC_CALLS_10DAYS','CAMP_CNT','DAYS_CAMP','AGE','ICR_RECH_CNT_60DAYS','USG_FLAG_04_10','CAMP_CNT_1','CAMP_CNT_2','CAMP_CNT_3','OG_ON_MOU_11_20oct','OG_OFF_MOU_11_20oct','SR_CNT_TOP1','SR_CNT_TOP2','ARPU_AUG17','ARPU_SEP17','ARPU_OCT17']])

#Model
#model= LogisticRegression(class_weight={0:.005, 1:.995})
#model= LogisticRegression()
#model = svm.SVC(kernel='linear', C=1, gamma=1) 
#model =DecisionTreeClassifier()
#model = RandomForestRegressor(n_estimator = 100, oob_score = TRUE, n_jobs = -1,random_state =50,max_features = "auto", min_samples_leaf = 50)
#model = RandomForestClassifier(n_estimators=100, min_samples_leaf = 100)
model = RandomForestClassifier(n_estimators=100)
X_train=dfTrain[features]
y_train=dfTrain.iloc[:,-1]
X_test=dfTest[features]
y_test=dfTest.iloc[:,-1]

model.fit(X_train,y_train)

##For persistance
#joblib.dump(model, 'model.pkl')
# Use clf = joblib.load('model.pkl') to call te same

pred_train=model.predict(X_train)
pred_test = model.predict(X_test)

#Concatinate

dfTemp["pred"]=pred_test
feature1=features+['PREDICTED_CHURN','pred','VLR_FLAG_21_27']
dfTemp[feature1].to_csv('out.csv')

#For decision Tree
#with open("a.dot", "w") as f:
#    f=export_graphviz(model,out_file=f)
#command = ["dot", "-Tpng", "a.dot", "-o", "a.png"]
#try:
#    subprocess.check_call(command)
#except:
#   print("Could not run dot, ie graphviz, to produce visualization")

#Error Analysis
accuracy_train = metrics.accuracy_score(pred_train,y_train)
accuracy_test = metrics.accuracy_score(pred_test,y_test)
confusion_matrix(y_test, pred_test)
F1=metrics.f1_score(y_test, pred_test)  
tn, fp, fn, tp = confusion_matrix(y_test, pred_test).ravel()

#Print
print("Training Accuracy=",accuracy_train,"and Test Accuracy=",accuracy_test)
print("tn, fp, fn, tp,precision,recall,F1,Accuracy")
print(tn,",", fp,",", fn,",", tp,",",round(tp/(tp+fp),4),",",round(tp/(tp+fn),4),",",round(F1,4),",",round(accuracy_test,4),",",round(time.clock()-start,4))