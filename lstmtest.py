import tensorflow
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, LSTM, Input
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

# show graphs of data
"""""
fig, ax = plt.subplots(5, figsize = (18,9))

ax[0].plot(dates, wat["level"])
ax[0].set_title("Watauga Level")
ax[0].set_ylabel("CFS")

ax[1].plot(dates, wat["precip"])
ax[1].set_title("Watauga Gauge Precip")
ax[1].set_ylabel("inches")

ax[2].plot(dates, be["precip"])
ax[2].set_title("Banner Elk Precip")
ax[2].set_ylabel("inches")

ax[3].plot(dates, vc["precip"])
ax[3].set_title("Valle Crucis Precip")
ax[3].set_ylabel("inches")

ax[4].plot(dates, bth["precip"])
ax[4].set_title("Bethel Precip")
ax[4].set_ylabel("inches")

plt.tight_layout()
plt.show()
"""""
# define lookback (number of days)
lookback = 30

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

fresh = og.values
fresh = np.reshape(fresh, (-1, 4))
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(fresh)
undoScaler = MinMaxScaler(feature_range=(0, 1))
fresh = np.reshape(fresh, (-1, 1))
undoScaler.fit(fresh)

# print(train)

# pre process the data
itrain = knnimp.transform(train)
itest = knnimp.transform(test)

daTrain = itrain.astype('float32')
daTrain = scaler.transform(daTrain)

daTest = itest.astype('float32')
daTest = scaler.transform(daTest)

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

# create matrices to feed to network
trainX, trainY = convert2matrix(daTrain, lookback)
testX, testY = convert2matrix(daTest, lookback)

print(trainX.shape)
#trainX = trainX.reshape(trainX.shape[0], trainX.shape[1], trainX.shape[2], 1)

# create that network!
daModel = tf.keras.models.Sequential()
#daModel.add(tf.keras.layers.Flatten())
#daModel.add(Input(shape=((lookback+1), 3)))
daModel.add(LSTM(lookback+1, input_shape=(lookback + 1, 3), activation='relu'))
#daModel.add(Dense((lookback+1) * 3, activation=tf.nn.relu))
daModel.add(Dense(1))

daModel.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])

daHistory = daModel.fit(trainX, trainY, epochs=100, batch_size=30, verbose=0, validation_data=(testX, testY),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=10)], shuffle=False)


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

#invert predictions
train_predict = undoScaler.inverse_transform(train_predict)
trainY = undoScaler.inverse_transform([trainY])
test_predict = undoScaler.inverse_transform(test_predict)
testY = undoScaler.inverse_transform([testY])

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
