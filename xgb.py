import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd

from sklearn.datasets import dump_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense

from numpy.random import seed
import xgboost

seed(0)
tf.random.set_seed(0)

# def build_model():
#     model = xgboost.XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=1, colsample_bytree=1)
#     return model

# def fine_tune(X_train, y_train, scaler_y, floor_area, total_price):
#     cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=0)
#     cvscores = []
#     for train, test in cv.split(X_train,y_train):
#         model = build_model()
#         result = model.fit(X_train[train], y_train[train], epochs=300, batch_size = int(len(X_train[train])/256), verbose=1, shuffle=False)
#         val_res = scaler_y.inverse_transform(model.predict(X_train[test]))
#         # print(val_res[0:10], labels[test][0:10])
#         # print(floor_area)
#         score = get_score(val_res * floor_area[test], total_price[test])
#         cvscores.append(score)
#     print('avarage score on cv = {}'.format(sum(cvscores)/len(cvscores)))
    

def get_mse_score(y_pred, y_val):
    return mean_squared_error(y_pred/100000, y_val/100000)

def get_mae_score(y_pred, y_val):
    return mean_squared_error(y_pred/100000, y_val/100000)

def get_rmse_score(y_pred, y_val):
    return mean_squared_error(y_pred/100000, y_val/100000, squared=False)

def main():
    # read data for train
    train = pd.read_csv('data_for_model/new_with_price_per_sqm/training_data.csv')
    LB = LabelBinarizer()
    train['town'] = LB.fit_transform(train['town']) 
    train['flat_model'] = LB.fit_transform(train['flat_model'])
    labels = train.iloc[:,20:].values
    total_price = train.iloc[:,19:20].values
    features = train.iloc[:,:19].values
    floor_area = np.asarray(train['floor_area_sqm'].values).reshape(len(labels),1)

    X_train = features
    y_train = labels
    train_dmatrix = xgboost.DMatrix(data = X_train, label = y_train)

    # read in test data
    test = pd.read_csv('data_for_model/new_with_price_per_sqm/test_data.csv')
    test['town'] = LB.fit_transform(test['town']) 
    test['flat_model'] = LB.fit_transform(test['flat_model'])
    labels_test = test.iloc[:,20:].values
    total_price_test = test.iloc[:,19:20].values
    features_test = test.iloc[:,:19].values
    floor_area_test = np.asarray(test['floor_area_sqm'].values).reshape(len(labels_test),1)

    X_test = features_test
    y_test = labels_test
    test_dmatrix = xgboost.DMatrix(data = X_test, label = y_test)

    # fine_tune
    # fine_tune(X_train, y_train, scaler_y, floor_area, total_price)

    # build model and train
    param = {"booster":"gblinear", "objective":"reg:linear"}
    model = xgboost.train(params = param, dtrain = train_dmatrix, num_boost_round = 500)
    # model = xgboost.XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
    print("finish building model")

    # get scores for validation
    y_valiadation = model.predict(train_dmatrix).reshape(len(labels),1)
    mae = get_mae_score(y_valiadation * floor_area, total_price)    
    print('mae score on validation = {}'.format(mae))
    mse = get_mse_score(y_valiadation * floor_area, total_price)
    print('mse score on validation = {}'.format(mse))
    rmse = get_rmse_score(y_valiadation * floor_area, total_price)
    print('rmse score on validation = {}'.format(rmse))
    
    # get score for test 
    y_pred = model.predict(test_dmatrix).reshape(len(labels_test),1)
    mae = get_mae_score(y_pred * floor_area_test, total_price_test)    
    print('mae score on test = {}'.format(mae))
    mse = get_mse_score(y_pred * floor_area_test, total_price_test)
    print('mse score on test = {}'.format(mse))
    rmse = get_rmse_score(y_pred * floor_area_test, total_price_test)
    print('rmse score on test = {}'.format(rmse))

    '''
    current best 
    1. on price/sqm
    
    2. on total price
    
    '''


if __name__ == "__main__":
    main()
