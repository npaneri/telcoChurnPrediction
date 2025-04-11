# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 19:01:44 2017

@author: nimish.paneri
"""
import pandas as pd
from matplotlib import pyplot as plt
dfTrain = pd.read_csv("G:\\Churn Project\\data5_new.csv")

Y=dfTrain['PREDICTED_CHURN']
X1=dfTrain['BALANCE']
X2=dfTrain['DAYS_RECHG']
plt.scatter(X2,X1,c=Y,marker='+')
#plt.scatter(Y,X2,color='blue')

#plt.figure(figsize=[16,12])

#plt.subplot(231)
#plt.hist()
#plt.boxplot(x=dfTrain['RECHG_CNT_10DAYS_2'], showmeans = True, meanline = True)
#plt.hist(dfTrain['LAST_USG_DAYS2'], 10, histtype = 'bar', facecolor = 'blue')
#plt.hist(x= dfTrain['DAYS_RECHG'])
#plt.hist(x = [dfTrain[dfTrain['PREDICTED_CHURN']==1]['LAST_USG_DAYS2'], dfTrain[dfTrain['PREDICTED_CHURN']==0]['LAST_USG_DAYS2']], 
#stacked=True, color = ['g','r'],label = ['CHURN','NOT CHURN'])
#plt.title('BALANCE histogram by Churn')
#plt.xlabel('x-axis')
#plt.ylabel('# predicted Churn')


# scatter plot of balance (x) and income (y)
#ax1 = plt.subplot(221)

plt.show()