import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder
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
    return scaler_x, scaler_x.transform(mtx)

def preprocessing_Y(mtx):
    scaler_y = StandardScaler()
    scaler_y.fit(mtx)
    return scaler_y, scaler_y.transform(mtx)

def build_model():
    model = Sequential()
    model.add(Dense(128, input_dim=63, kernel_initializer='normal',activation='selu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='orthogonal', activation='linear'))    
    model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
    return model

def build_simple_model():
    model = Sequential()
    model.add(Dense(32, input_dim=63, kernel_initializer='normal',activation='relu'))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='linear'))    
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
    # read data
    train = pd.read_csv('data_for_model/new_with_price_per_sqm/training_data.csv')
    test = pd.read_csv('data_for_model/new_with_price_per_sqm/test_data.csv')
    
    LE1 = LabelEncoder()
    train['town'] = LE1.fit_transform(train['town']) 
    test['town'] = LE1.transform(test['town']) 

    LE2 = LabelEncoder()
    train['flat_model'] = LE2.fit_transform(train['flat_model'])
    test['flat_model'] = LE2.transform(test['flat_model'])

    OHE1 = OneHotEncoder(handle_unknown='ignore')
    l1 = pd.DataFrame(OHE1.fit_transform(train[['town']]).toarray())
    train = train.join(l1)
    l2 = pd.DataFrame(OHE1.transform(test[['town']]).toarray())
    test = test.join(l2)

    OHE2 = OneHotEncoder(handle_unknown='ignore')
    l3 = pd.DataFrame(OHE2.fit_transform(train[['flat_model']]).toarray())
    l3.columns = list(range(26,46))
    train = train.join(l3)
    l4 = pd.DataFrame(OHE2.transform(test[['flat_model']]).toarray())
    l4.columns = list(range(26,46))
    test = test.join(l4)

    labels = train['price_per_sqm'].values
    labels = np.asarray(labels).reshape(len(labels),1)
    total_price = np.asarray(train['resale_price'].values).reshape(len(labels),1)
    features = train.drop(columns=['town', 'flat_model','resale_price','price_per_sqm'])
    floor_area = np.asarray(train['floor_area_sqm'].values).reshape(len(labels),1)

    # preprocess training data
    scaler_x, X_train = preprocessing_X(features)
    scaler_y, y_train = preprocessing_Y(labels)

    # read in test data
    
    labels_test = test['price_per_sqm'].values
    labels_test = np.asarray(labels_test).reshape(len(labels_test),1)
    total_price_test = np.asarray(test['resale_price'].values).reshape(len(labels_test),1)
    features_test = test.drop(columns=['town', 'flat_model','resale_price','price_per_sqm'])
    floor_area_test = np.asarray(test['floor_area_sqm'].values).reshape(len(labels_test),1)

    # preprocess test data
    X_test = scaler_x.transform(features_test)
    y_test = scaler_y.transform(labels_test)

    # fine_tune
    # fine_tune(X_train, y_train, scaler_y, floor_area, total_price)
    
    # train on all training data with best hyper-params
    # model = build_model()
    model = build_simple_model()
    # result = model.fit(X_train, y_train, epochs=300, batch_size=256, verbose=1, shuffle=False)
    result = model.fit(X_train, y_train, epochs=100)

    # get scores for validation
    # y_valiadation = model.predict(train_dmatrix).reshape(len(labels),1)
    y_valiadation = scaler_y.inverse_transform(model.predict(X_train))
    mae = get_mae_score(y_valiadation * floor_area, total_price)
    print('mae score on validation = {}'.format(mae))
    mse = get_mse_score(y_valiadation * floor_area, total_price)
    print('mse score on validation = {}'.format(mse))
    rmse = get_rmse_score(y_valiadation * floor_area, total_price)
    print('rmse score on validation = {}'.format(rmse))
    # mae = get_mae_score(y_valiadation, total_price)
    # print('mae score on validation = {}'.format(mae))
    # mse = get_mse_score(y_valiadation, total_price)
    # print('mse score on validation = {}'.format(mse))
    # rmse = get_rmse_score(y_valiadation, total_price)
    # print('rmse score on validation = {}'.format(rmse))
    
    # get score for test 
    y_pred = scaler_y.inverse_transform(model.predict(X_test))
    mae = get_mae_score(y_pred * floor_area_test, total_price_test)    
    print('mae score on test = {}'.format(mae))
    mse = get_mse_score(y_pred * floor_area_test, total_price_test)
    print('mse score on test = {}'.format(mse))
    rmse = get_rmse_score(y_pred * floor_area_test, total_price_test)
    print('rmse score on test = {}'.format(rmse))
    # mae = get_mae_score(y_pred, total_price_test)    
    # print('mae score on test = {}'.format(mae))
    # mse = get_mse_score(y_pred, total_price_test)
    # print('mse score on test = {}'.format(mse))
    # rmse = get_rmse_score(y_pred, total_price_test)
    # print('rmse score on test = {}'.format(rmse))
    

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
