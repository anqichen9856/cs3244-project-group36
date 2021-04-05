import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from keras.models import Sequential
from keras.layers import Dense

from numpy.random import seed
from tensorflow import set_random_seed

seed(0)
set_random_seed(0)

def preprocessing_X(mtx):
    scaler_x = MinMaxScaler()
    scaler_x.fit(mtx)
    return scaler_x.transform(mtx)

def preprocessing_Y(mtx):
    scaler_y = MinMaxScaler()
    scaler_y.fit(mtx)
    return scaler_y, scaler_y.transform(mtx)

def build_model():
    model = Sequential()
    model.add(Dense(19, input_dim=19, kernel_initializer='normal',activation='relu'))
    model.add(Dense(8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='linear'))    
    model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
    return model

def train_model(X, y, model):
    # param_grid = dict(filters=[256, 512, 1024],
    #               kernel_size=[2, 3],
    #               pool_size=[2, 3],
    #               dropout_rate=[0.15, 0.2], 
    #               n_dense_1=[512], 
    #               n_dense_2=[256], 
    #               n_dense_3=[128])
    model.fit(X, y, epochs=50, batch_size = len(X), verbose=1, shuffle=False)

def predict(features, model):
    sc = StandardScaler()
    X = sc.fit_transform(features)
    return model.predict(X)

def get_score(y_pred, y_val):
    return mean_squared_error(y_pred/100000, y_val/100000)

def main():
    # read data for train
    train = pd.read_csv('data_for_model/training_data.csv')
    LE = LabelEncoder()
    train['town'] = LE.fit_transform(train['town']) 
    train['flat_model'] = LE.fit_transform(train['flat_model'])
    labels = train.iloc[:10000,19:20].values
    features = train.iloc[:10000,:19].values

    # preprocess training data
    X_train = preprocessing_X(features)
    scaler_y, y_train = preprocessing_Y(labels)

    # build and train model
    model = build_model()
    train_model(X_train, y_train, model)
    
    # read in validation data
    validation = pd.read_csv('data_for_model/validation_data.csv')
    validation['town'] = LE.fit_transform(validation['town']) 
    validation['flat_model'] = LE.fit_transform(validation['flat_model'])
    labels_val = validation.iloc[:,19:20].values
    features_val = validation.iloc[:,:19].values

    # preprocess validation data
    X_val = preprocessing_X(features_val)
    scaler_y, y_val = preprocessing_Y(labels_val)

    # predict y values for validation data
    val_res = scaler_y.inverse_transform(predict(X_val, model))

    # get performance score on validation data
    score = get_score(val_res, labels_val)
    print('score on validation = {}'.format(score))

if __name__ == "__main__":
    main()
