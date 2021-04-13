#!/usr/bin/env python
# coding: utf-8

# In[33]:
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from time import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense

from numpy.random import seed

from sklearn.inspection import plot_partial_dependence
from keras.wrappers.scikit_learn import KerasRegressor

df_train = pd.read_csv('data_for_model/new_with_price_per_sqm/training_data.csv')
df_test =  pd.read_csv('data_for_model/new_with_price_per_sqm/test_data.csv')


# In[36]:


df_train.columns


# In[39]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
enc1 = LabelEncoder()
df_train['flat_model'] = enc1.fit_transform(df_train['flat_model'])
df_test['flat_model'] = enc1.transform(df_test['flat_model'])
enc2 = OneHotEncoder(handle_unknown='ignore')
enc_df = pd.DataFrame(enc2.fit_transform(df_train[['flat_model']]).toarray())
df_train = df_train.join(enc_df)
enc_df = pd.DataFrame(enc2.transform(df_test[['flat_model']]).toarray())
df_test = df_test.join(enc_df)


# In[45]:


X = df_train.drop(columns=['town', 'flat_model','resale_price','price_per_sqm'])
y_1 = df_train['resale_price']
y_2 = df_train['price_per_sqm']
X_test = df_test.drop(columns=['town', 'flat_model','resale_price','price_per_sqm'])
y_test_1 = df_test['resale_price']
y_test_2 = df_test['price_per_sqm']


# In[46]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold

from xgboost import XGBClassifier

print(1)
'''
regressor_1 = RandomForestRegressor(n_estimators=20, random_state=0)
regressor_1.fit(X, y_1)

# CV 
kfold = KFold(n_splits=5, shuffle=True)
kf_cv_scores = cross_val_score(regressor_1, X, y_1, cv=kfold, scoring='neg_mean_squared_error')
print("K-fold CV average MSE: %.2f" % kf_cv_scores.mean())

y_pred_1 = regressor_1.predict(X_test)


# In[52]:


from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_1/100000, y_pred_1/100000))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_1/100000, y_pred_1/100000))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_1/100000, y_pred_1/100000)))
'''

# In[48]:
'''
xgb_model = XGBClassifier()
xgb_model.fit(X, y_2)
'''
regressor_2 = RandomForestRegressor(n_estimators=20, random_state=0)
regressor_2.fit(X, y_2)

print('Computing partial dependence plots...')
tic = time()
fig, ax = plt.subplots(figsize=(12, 6))

pdp_features = [4] # remaining_lease, dist_nearest_mrt, dist_nearest_supermarkets, dist_nearest_sports_facilities

X_pdp = df_train.loc[df_train['town']=="ANG MO KIO"].drop(columns=['town', 'flat_model','resale_price','price_per_sqm'])
display1 = plot_partial_dependence(estimator=regressor_2, X=X_pdp, features=pdp_features,
        kind='both', subsample=50, random_state=0, verbose=10)#, method='recursion')
display1.plot(ax=[ax], line_kw={"label": "ANG MO KIO", "color": "red"})

X_pdp = df_train.loc[df_train['town']=="SENGKANG"].drop(columns=['town', 'flat_model','resale_price','price_per_sqm'])
display2 = plot_partial_dependence(estimator=regressor_2, X=X_pdp, features=pdp_features,
        kind='both', subsample=50, random_state=0, verbose=10)#, method='recursion')
display2.plot(ax=[ax], line_kw={"label": "SENGKANG", "color": "blue"})
print(f"done in {time() - tic:.3f}s")

X_pdp = df_train.loc[df_train['town']=="BUKIT TIMAH"].drop(columns=['town', 'flat_model','resale_price','price_per_sqm'])
display3 = plot_partial_dependence(estimator=regressor_2, X=X_pdp, features=pdp_features,
        kind='both', subsample=50, random_state=0, verbose=10)#, method='recursion')
display3.plot(ax=[ax], line_kw={"label": "BUKIT TIMAH", "color": "green"})
print(f"done in {time() - tic:.3f}s")


X_pdp = df_train.loc[df_train['#rooms']>=5].drop(columns=['town', 'flat_model','resale_price','price_per_sqm'])
display4 = plot_partial_dependence(estimator=regressor_2, X=X_pdp, features=pdp_features,
        kind='both', subsample=50, random_state=0, verbose=10)#, method='recursion')
display4.plot(ax=[ax], line_kw={"label": "Large Flats (>=5rm)", "color": "purple"})
print(f"done in {time() - tic:.3f}s")

ax.set_ylim([3000, 8000])


'''
display.figure_.suptitle(
    'Plot'
)
display.figure_.subplots_adjust(hspace=0.3)
'''
plt.show()

# CV 
kfold = KFold(n_splits=5, shuffle=True)
kf_cv_scores = cross_val_score(regressor_2, X, y_2, cv=kfold, scoring='neg_mean_squared_error')
print("K-fold CV average MSE: %.2f" % kf_cv_scores.mean())

y_pred_2 = regressor_2.predict(X_test)*X_test['floor_area_sqm']


# In[51]:


from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_1/100000, y_pred_2/100000))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_1/100000, y_pred_2/100000))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_1/100000, y_pred_2/100000)))


# In[ ]:




