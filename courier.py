import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb 
sb.set_style("whitegrid", {'axes.grid' : False})
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier


lifetime = pd.read_csv('Courier_lifetime_data.csv')
weekly = pd.read_csv('Courier_weekly_data.csv')

plt.hist(lifetime.feature_2)



# keep only data within +3 to -3 standard deviations 
outliers_ix = lifetime.index[np.abs(lifetime.feature_2-lifetime.feature_2.mean()) >= (3*lifetime.feature_2.std())]
lifetime.loc[outliers_ix, 'feature_2'] = np.NaN


#1st find couriers that do have weekly data to inspect the distributions
merge = pd.merge(lifetime, weekly, on='courier', how='right') #.to_csv('merge.csv')

#Then get the complete dataset. Lots of couriers will have lifetime data but no weekly data:
merge_all = pd.merge(lifetime, weekly, on='courier', how='left') #.to_csv('merge.csv')


#Plot distributions to chech whether lifetime categories a, b, c, d are related to the distribution of the weekly features

a = merge.loc[merge.feature_1_x=='a']
b = merge.loc[merge.feature_1_x=='b']
c = merge.loc[merge.feature_1_x=='c']
d = merge.loc[merge.feature_1_x=='d']

weekly_features = ['feature_1_y', 'feature_2_y', 'feature_3',
       'feature_4', 'feature_5', 'feature_6', 'feature_7',
       'feature_8', 'feature_9', 'feature_10', 'feature_11',
       'feature_12', 'feature_13', 'feature_14', 'feature_15',
       'feature_16', 'feature_17']

fig, axes = plt.subplots(9, 2, figsize=(20,30))

ax = axes.flatten() #ravel()

for i in range(len(weekly_features)):
    ax[i].hist(a.ix[:,weekly_features[i]], bins=20, color='red', alpha=.5)
    ax[i].hist(b.ix[:,weekly_features[i]], bins=20, color='green', alpha=.5)
    ax[i].hist(c.ix[:,weekly_features[i]], bins=20, color='blue', alpha=.5)
    ax[i].hist(d.ix[:,weekly_features[i]], bins=20, color='yellow', alpha=.5)
    ax[i].set_title(weekly_features[i], fontsize=18)
    ax[i].set_yticks(())
    
ax[0].set_xlabel("Feature distribution")
ax[0].legend(["a", "b", "c", "d"], loc="best")
fig.tight_layout()

plt.show()








#First knn-predict lifetime.feature_2 . I can only do this for couriers with weekly features 
#First create train X and target y
'''
X = np.matrix(merge.dropna()[weekly_features]) #will remove rows with missing feature_2_x 
y = np.matrix(merge['feature_2_x'].dropna())

#k = 5 by default
regr = KNeighborsRegressor().fit(X, y.transpose())
feature_2_x_ = regr.predict(np.matrix(merge[weekly_features]))#[:, 'feature_2_x']
merge['feature_2_x_'] = [int(i) for i in feature_2_x_]
'''

'''
#Do the same knn-prediction of feature_2_x for subsets a, b, c, d:
X = np.matrix(a.dropna()[weekly_features])
y = np.matrix(a['feature_2_x'].dropna())

#k = 5 by default
regr = KNeighborsRegressor().fit(X, y.transpose())
feature_2_x_ = regr.predict(np.matrix(a[weekly_features]))#[:, 'feature_2_x']
a['feature_2_x_'] = [int(i) for i in feature_2_x_]
'''

'''

#Replace the missing values in feature_2_x with the knn-predicted feature_2_x_ values
missing_f2_ix = merge.index[merge.feature_2_x.isnull()]
predicted_f2  = merge[merge.feature_2_x.isnull()].feature_2_x_
merge.loc[missing_f2_ix, 'feature_2_x'] = predicted_f2

#Note: the predicted feature_2_x values will be different for same courier! Is this right?

#Find the couriers in merge_all and replace the missing feature_2 values with the predicted one
#(will only affect the missing feature_2_x for the couriers that have weekly data)
merge.loc[missing_f2_ix, 'courier']

'''










#Impute missing values with combinations of feature_1_x and feature_2_x
#For example, of 103 couriers with feature1: a, feature2: 31.0 
#43 of them have weekly data. Impute the remaining 60 ones with the median of the 43

#Note I can impute weekly features like this.
#But not week, since it's not a continuous variable

list_f1 = list(set(merge_all.feature_1_x))
list_f2 = list(set(merge_all[merge_all.feature_2_x.notnull()].feature_2_x))

frames = []
for f1 in list_f1:
  for f2 in list_f2:
    df = merge_all.loc[merge_all.feature_1_x==f1][merge_all.feature_2_x==f2]
    if len(df.dropna())>0: #some feature 1, feature 2 combinations don't have weekly data
      df = df.fillna(df[weekly_features].dropna().median())
      frames.append(df)
    else:
      df = df.fillna(merge_all.loc[merge_all.feature_1_x==f1].dropna().median())
      frames.append(df)
#And then for the 1227 couriers that have feature_1 but no feature_2 just impute the median of the category a, b, c, d
for f1 in list_f1:
  df = merge_all[merge_all.feature_2_x.isnull()][merge_all.feature_1_x==f1]
  df = df.fillna(df[weekly_features].dropna().median())
  frames.append(df)
data = pd.concat(frames)





#With the imputed data I can now knn-predict the missing feature_2_x values
X = np.matrix(data[data['feature_2_x'].notnull()][weekly_features])
y = np.matrix(data['feature_2_x'].dropna()).T

#k = 5 by default
regr = KNeighborsRegressor().fit(X, y)
feature_2_x_ = regr.predict(np.matrix(data[weekly_features]))#[:, 'feature_2_x']
data['feature_2_x_'] = feature_2_x_ #[int(i) for i in feature_2_x_]
#Replace the missing values in feature_2_x with the knn-predicted feature_2_x_ values
missing_f2_ix = data.index[data.feature_2_x.isnull()]
predicted_f2  = data[data.feature_2_x.isnull()].feature_2_x_
data.loc[missing_f2_ix, 'feature_2_x'] = predicted_f2


#Do not predict week. Couriers with missing values for weeks 9, 10 and 11 will be removed
'''
X = np.matrix(data[data.week.notnull()][weekly_features])
y = np.matrix(data.week.dropna()).T
clf = KNeighborsClassifier().fit(X,y.ravel().tolist()[0])
week_ = clf.predict(np.matrix(data[weekly_features]))#[:, 'feature_2_x']
data['week_'] = week_
missing_week_ix = data.index[data.week.isnull()]
predicted_week  = data[data.week.isnull()].week_
data.loc[missing_week_ix, 'week'] = predicted_week
'''


sb.pairplot(data[weekly_features], kind='reg')
plt.show()