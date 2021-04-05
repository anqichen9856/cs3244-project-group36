import numpy as np
from numpy.random import seed
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from keras.models import Sequential
from keras.layers import Dense
from sklearn.inspection import plot_partial_dependence
from keras.wrappers.scikit_learn import KerasRegressor

def preprocessing_X(mtx):
    scaler_x = MinMaxScaler()
    scaler_x.fit(mtx)
    return scaler_x.transform(mtx)

def preprocessing_Y(mtx):
    scaler_y = MinMaxScaler()
    scaler_y.fit(mtx)
    return scaler_y, scaler_y.transform(mtx)

def train_model(X, y, model):
    model.fit(X, y, epochs=10, steps_per_epoch=100)

def predict(features, model):
    sc = StandardScaler()
    X = sc.fit_transform(features)
    return model.predict(X)

def get_score(y_pred, y_val):
    return mean_squared_error(y_pred/100000, y_val/100000)

def build_network():
    model = Sequential()
    model.add(Dense(19, input_dim=19, activation='relu'))
    model.add(Dense(2000, activation='relu'))
    model.add(Dense(1, activation='linear'))    
    model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
    return model

def main():
    seed(0)
    tf.random.set_seed(0)

    train = pd.read_csv('data_for_model/training_data.csv')
    LE = LabelEncoder()
    train['town'] = LE.fit_transform(train['town']) 
    train['flat_model'] = LE.fit_transform(train['flat_model'])
    labels = train.iloc[:5000,19:20].values
    features = train.iloc[:5000,:19].values

    X_train = preprocessing_X(features)
    scaler_y, y_train = preprocessing_Y(labels)

    '''
    model = Sequential()
    model.add(Dense(19, input_dim=19, activation='relu'))
    model.add(Dense(2000, activation='relu'))
    model.add(Dense(1, activation='linear'))    
    model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

    
    train_model(X_train, y_train, model)
    
    validation = pd.read_csv('data_for_model/validation_data.csv')
    validation['town'] = LE.fit_transform(validation['town']) 
    validation['flat_model'] = LE.fit_transform(validation['flat_model'])
    labels_val = validation.iloc[:,19:20].values
    features_val = validation.iloc[:,:19].values

    X_val = preprocessing_X(features_val)
    scaler_y, y_val = preprocessing_Y(labels_val)

    val_res = scaler_y.inverse_transform(predict(X_val, model))
    score = get_score(val_res, labels_val)
    print('score on validation = {}'.format(score))
    '''

    model = KerasRegressor(build_network, epochs=10, steps_per_epoch=100)

    model._estimator_type = "regressor" # Cheap workaround for keras NN to work with plot_partial_dependence
    model.dummy_ = "dummy" # Cheap workaround for keras NN to work with plot_partial_dependence

    model.fit(X_train, y_train)

    print('Computing partial dependence plots...')
    tic = time()
    pdp_features = [5, 10, 16, 18] # remaining_lease, dist_nearest_mrt, dist_nearest_supermarkets, dist_nearest_sports_facilities
    display = plot_partial_dependence(estimator=model, X=X_train, features=pdp_features,
        kind='average', subsample=25, random_state=0, verbose=10)
    print(f"done in {time() - tic:.3f}s")

    display.figure_.suptitle(
    'Plot'
    )
    display.figure_.subplots_adjust(hspace=0.3)
    plt.show()

if __name__ == "__main__":
    main()

