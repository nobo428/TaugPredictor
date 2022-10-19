import tensorflow
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

# import data from csv files
be = pd.read_csv(r'csv/bannerelk.csv')
vc = pd.read_csv(r'csv/vallecrucis.csv')
bth = pd.read_csv(r'csv/bethel.csv')
wat = pd.read_csv(r'csv/watauga.csv')
dates = pd.to_datetime(wat["datetime"], format="%Y%m%d")

# combine multiple dataframes into one with readable columns
og = pd.concat([be["precip"], vc["precip"], bth["precip"], wat["level"]], axis=1)
og.index = dates
og.columns = ["be_precip", "vc_precip", "bth_precip", "level"]

# print(og)

# train imputer
knnimp = KNNImputer().fit(og)

# split data to train and test sets
train = og[og.index < pd.to_datetime("2022-01-01", format='%Y-%m-%d')]
test = og[og.index >= pd.to_datetime("2022-01-01", format='%Y-%m-%d')]

# print(train)

# pre process the data
itrain = knnimp.transform(train)
itest = knnimp.transform(test)

# display graph of train/test split
"""""
plt.plot(train.index, itrain[:, 3], color="black")
plt.plot(test.index, itest[:, 3], color="red")
plt.ylabel('level')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.title("Train/Test split for Level Data")
plt.show()
"""""

# helper for getting training data
def convert2matrix(data, look_back):
    X, Y = [], []
    for i in range(len(data) - look_back):
        d = i + look_back
        X.append(data[i:d + 1, 0:3])
        Y.append(data[d, 3])
    return np.array(X), np.array(Y)


# define lookback (number of days)
lookback = 60

# create matrices to feed to network
trainX, trainY = convert2matrix(itrain, lookback)
testX, testY = convert2matrix(itest, lookback)

# create that network!
daModel = tf.keras.models.Sequential()
daModel.add(tf.keras.layers.BatchNormalization())
daModel.add(tf.keras.layers.Flatten())
daModel.add(Input(shape=(lookback+1, 3)))
daModel.add(Dense((lookback+1) * 3, activation=tf.nn.relu))
#daModel.add(Dense((lookback+1) * 6, activation=tf.nn.sigmoid))
daModel.add(Dense((lookback+1) * 6, activation=tf.nn.relu))
#daModel.add(Dense((lookback+1) * 6, activation=tf.nn.tanh))
daModel.add(Dense((lookback+1) * 3, activation=tf.nn.relu))
daModel.add(Dense(1))

daModel.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])

daHistory = daModel.fit(trainX, trainY, epochs=100, batch_size=256, verbose=False, validation_data=(testX, testY),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=10)], shuffle=True)


def model_loss(history):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(loc='upper right')
    plt.show()

train_predict = daModel.predict(trainX)
test_predict = daModel.predict(testX)
#print(testY.size)
#print(test_predict.size)

train_score = daModel.evaluate(trainX, trainY, verbose=0)
print('Train Root Mean Squared Error(RMSE): %.2f; Train Mean Absolute Error(MAE) : %.2f '
      % (np.sqrt(train_score[1]), train_score[2]))

test_score = daModel.evaluate(testX, testY, verbose=0)
print('Test Root Mean Squared Error(RMSE): %.2f; Test Mean Absolute Error(MAE) : %.2f '
      % (np.sqrt(test_score[1]), test_score[2]))

model_loss(daHistory)

def train_prediction_plot(testY, test_predict):
    plt.figure(figsize=(8, 4))
    plt.plot(train.index[lookback:], testY, marker='.', label="actual")
    plt.plot(train.index[lookback:], test_predict, 'r', label="prediction")
    plt.tight_layout()
    plt.subplots_adjust(left=0.07)
    plt.ylabel('Level (cfs)', size=15)
    plt.xlabel('Date', size=15)
    plt.legend(fontsize=15)
    plt.show()


def prediction_plot(testY, test_predict):
    plt.figure(figsize=(8, 4))
    plt.plot(test.index[lookback:], testY, marker='.', label="actual")
    plt.plot(test.index[lookback:], test_predict, 'r', label="prediction")
    plt.tight_layout()
    plt.subplots_adjust(left=0.07)
    plt.ylabel('Level (cfs)', size=15)
    plt.xlabel('Date', size=15)
    plt.legend(fontsize=15)
    plt.show()

#train_prediction_plot(trainY, train_predict)
prediction_plot(testY, test_predict)

#commented out to not re-save while working on predictor
#daModel.save('taugModel')

# print(be)
# print(vc)
# print(bth)
# print(wat)
# print(mf)
