import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense

from numpy.random import seed
from tensorflow import set_random_seed

seed(0)
set_random_seed(0)

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
    model.add(Dense(19, input_dim=19, kernel_initializer='normal',activation='relu'))
    model.add(Dense(8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='linear'))    
    model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
    return model

def predict(features, model):
    # sc = StandardScaler()
    # X = sc.fit_transform(features)
    return model.predict(features)

def get_score(y_pred, y_val):
    return mean_squared_error(y_pred/100000, y_val/100000)

def main():
    # read data for train
    train = pd.read_csv('data_for_model/training_data.csv')
    LE = LabelEncoder()
    train['town'] = LE.fit_transform(train['town']) 
    train['flat_model'] = LE.fit_transform(train['flat_model'])
    labels = train.iloc[:,19:20].values
    features = train.iloc[:,:19].values

    # preprocess training data
    X_train = preprocessing_X(features)
    scaler_y, y_train = preprocessing_Y(labels)

    # build and train model
    # model = KerasClassifier(build_fn=build_model, verbose=1)
    # param_grid = dict(epochs=[20,50,100],
    #               batch_size=[1000,5000,len(X_train)],
    #             #   solver=['svd', 'cholesky', 'lsqr', 'sag'],
    #               optimizer= ['rmsprop', 'adam'],
    #               init= ['glorot_uniform', 'normal', 'uniform']
    #             #   penalty=['l1', 'l2']
    #             )
    cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=0)
    cvscores = []
    # search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_root_mean_squared_error', n_jobs=-1, cv=cv)
    for train, test in cv.split(X_train,y_train):
        model = build_model()
        result = model.fit(X_train[train], y_train[train], epochs=100, batch_size = len(X_train[train]), verbose=1, shuffle=False)
        val_res = scaler_y.inverse_transform(predict(X_train[test], model))
        print(val_res[0:10], labels[test][0:10])
        score = get_score(val_res, labels[test])
        cvscores.append(score)
    
    print(cvscores)

    # summarize result
    # print('Best Score: %s' % result.best_score_)
    # print('Best Hyperparameters: %s' % result.best_params_)
    
    # read in validation data
    # validation = pd.read_csv('data_for_model/validation_data.csv')
    # validation['town'] = LE.fit_transform(validation['town']) 
    # validation['flat_model'] = LE.fit_transform(validation['flat_model'])
    # labels_val = validation.iloc[:,19:20].values
    # features_val = validation.iloc[:,:19].values

    # # preprocess validation data
    # X_val = preprocessing_X(features_val)
    # scaler_y, y_val = preprocessing_Y(labels_val)

    # # predict y values for validation data
    # val_res = scaler_y.inverse_transform(predict(X_val, search))

    # # get performance score on validation data
    # score = get_score(val_res, labels_val)
    # print('score on validation = {}'.format(score))

if __name__ == "__main__":
    main()

