import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import RepeatedKFold,GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense

from numpy.random import seed

seed(0)
tf.random.set_seed(0)

def preprocessing_X(mtx):
    scaler_x = StandardScaler()
    scaler_x.fit(mtx)
    return scaler_x.transform(mtx)

def preprocessing_Y(mtx):
    scaler_y = StandardScaler()
    scaler_y.fit(mtx)
    return scaler_y, scaler_y.transform(mtx)

def build_model():
    model = Sequential()
    model.add(Dense(100, input_dim=19, kernel_initializer='normal',activation='selu'))
    model.add(Dense(90, kernel_initializer='normal', activation='relu'))
    model.add(Dense(80, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='orthogonal', activation='linear'))    
    model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
    return model

# def fine_tune(X_train, y_train, scaler_y, floor_area, total_price):
#     cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=0)
#     cvscores = []
#     for train, test in cv.split(X_train,y_train):
#         model = build_model()
#         result = model.fit(X_train[train], y_train[train], epochs=500, batch_size = int(len(X_train[train])/256), verbose=1, shuffle=False)
#         val_res = scaler_y.inverse_transform(model.predict(X_train[test]))
#         # print(val_res[0:10], labels[test][0:10])
#         # print(floor_area)
#         score = get_score(val_res * floor_area[test], total_price[test])
#         cvscores.append(score)
#     print('avarage score on cv = {}'.format(sum(cvscores)/len(cvscores)))
    
    # for train, test in cv.split(X_train,y_train):
    #     model = build_model()
    #     result = model.fit(X_train[train], y_train[train], epochs=100, batch_size = len(X_train[train]), verbose=1, shuffle=False)
    #     val_res = scaler_y.inverse_transform(predict(X_train[test], model))
    #     print(val_res[0:10], labels[test][0:10])
    #     score = get_score(val_res, labels[test])
    #     cvscores.append(score)
    

def get_mse_score(y_pred, y_val):
    return mean_squared_error(y_pred/10000, y_val/10000)

def get_mae_score(y_pred, y_val):
    return mean_absolute_error(y_pred/10000, y_val/10000)

def get_rmse_score(y_pred, y_val):
    return mean_squared_error(y_pred/10000, y_val/10000, squared=False)

def main():
    # read data for train
    train = pd.read_csv('data_for_model/new_with_price_per_sqm/training_data.csv')
    LE = LabelEncoder()
    train['town'] = LE.fit_transform(train['town']) 
    train['flat_model'] = LE.fit_transform(train['flat_model'])
    labels = train.iloc[:,20:].values
    total_price = train.iloc[:,19:20].values
    features = train.iloc[:,:19].values
    floor_area = np.asarray(train['floor_area_sqm'].values).reshape(len(labels),1)

    # preprocess training data
    X_train = preprocessing_X(features)
    scaler_y_train, y_train = preprocessing_Y(labels)

    # read in test data
    test = pd.read_csv('data_for_model/new_with_price_per_sqm/test_data.csv')
    test['town'] = LE.fit_transform(test['town']) 
    test['flat_model'] = LE.fit_transform(test['flat_model'])
    labels_test = test.iloc[:,20:].values
    total_price_test = test.iloc[:,19:20].values
    features_test = test.iloc[:,:19].values
    floor_area_test = np.asarray(test['floor_area_sqm'].values).reshape(len(labels_test),1)

    # preprocess test data
    X_test = preprocessing_X(features_test)
    scaler_y_test, y_test = preprocessing_Y(labels_test)

    # fine_tune
    # fine_tune(X_train, y_train, scaler_y, floor_area, total_price)
    
    # train on all training data with best hyper-params
    model = build_model()
    result = model.fit(X_train, y_train, epochs=400, batch_size = int(len(X_train)/256), verbose=1, shuffle=False)

    # get scores for validation
    # y_valiadation = model.predict(train_dmatrix).reshape(len(labels),1)
    y_valiadation = scaler_y_train.inverse_transform(model.predict(X_train))
    mae = get_mae_score(y_valiadation * floor_area, total_price)    
    print('mae score on validation = {}'.format(mae))
    mse = get_mse_score(y_valiadation * floor_area, total_price)
    print('mse score on validation = {}'.format(mse))
    rmse = get_rmse_score(y_valiadation * floor_area, total_price)
    print('rmse score on validation = {}'.format(rmse))
    
    # get score for test 
    y_pred = scaler_y_test.inverse_transform(model.predict(X_test))
    mae = get_mae_score(y_pred * floor_area_test, total_price_test)    
    print('mae score on test = {}'.format(mae))
    mse = get_mse_score(y_pred * floor_area_test, total_price_test)
    print('mse score on test = {}'.format(mse))
    rmse = get_rmse_score(y_pred * floor_area_test, total_price_test)
    print('rmse score on test = {}'.format(rmse))
    

    '''
    current best 
    1. on price/sqm
    score on validation = 0.13425226463511938
    score on test = 0.13700108057884386 
    2. on total price
    score on validation = 0.13950871271685533
    score on test = 0.14417668489260577
    '''


if __name__ == "__main__":
    main()
