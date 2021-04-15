#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df_train = pd.read_csv('training_data.csv')
df_test =  pd.read_csv('test_data.csv')


# In[2]:


df_train.columns


# In[3]:


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


# In[4]:


enc1 = LabelEncoder()
df_train['town'] = enc1.fit_transform(df_train['town'])
df_test['town'] = enc1.transform(df_test['town'])
enc2 = OneHotEncoder(handle_unknown='ignore')
enc_df = pd.DataFrame(enc2.fit_transform(df_train[['town']]).toarray())
enc_df.columns = list(range(20,46))
df_train = df_train.join(enc_df)
enc_df = pd.DataFrame(enc2.transform(df_test[['town']]).toarray())
enc_df.columns = list(range(20,46))
df_test = df_test.join(enc_df)


# In[5]:


X = df_train.drop(columns=['town', 'flat_model','resale_price','price_per_sqm'])
y_1 = df_train['resale_price']
y_2 = df_train['price_per_sqm']
X_test = df_test.drop(columns=['town', 'flat_model','resale_price','price_per_sqm'])
y_test_1 = df_test['resale_price']
y_test_2 = df_test['price_per_sqm']


# In[6]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold

regressor_1 = RandomForestRegressor(n_estimators=20, random_state=0)
regressor_1.fit(X, y_1)

# CV 
kfold = KFold(n_splits=5, shuffle=True)
kf_cv_scores = cross_val_score(regressor_1, X, y_1, cv=kfold, scoring='neg_mean_squared_error')
print("K-fold CV average MSE: %.2f" % kf_cv_scores.mean())

y_pred_1 = regressor_1.predict(X_test)


# In[7]:


from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_1/100000, y_pred_1/100000))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_1/100000, y_pred_1/100000))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_1/100000, y_pred_1/100000)))


# In[48]:


regressor_2 = RandomForestRegressor(n_estimators=20, random_state=0)
regressor_2.fit(X, y_2)

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


# In[57]:


# Parameter Tuning
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = {'bootstrap': [True, False],
 'max_depth': [20,40, 60,80, 100, None],
 'max_features': ['auto', 'sqrt'],
 'n_estimators': [10, 20,50]}

regressor = RandomForestRegressor()# Instantiate the grid search model
rf_random = RandomizedSearchCV(estimator = regressor, param_distributions = param_grid, n_iter = 30, random_state=22,
                          cv = 5, n_jobs = 1, verbose = 2)
rf_random.fit(X, y_2)


# In[65]:


regressor = rf_random.best_estimator_
y_pred_2 = regressor.predict(X_test)*X_test['floor_area_sqm']
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_1/100000, y_pred_2/100000))
print('Mean Squared Error:', metrics.mean_squared_error(y_test_1/100000, y_pred_2/100000))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_1/100000, y_pred_2/100000)))


# In[ ]:




