import pandas as pd

import csv
import codecs
import urllib.request
import urllib.error
import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from io import StringIO

#define lookback and forecast windows
lookback = 60
forecast = 10

#getting the correct dates
today = datetime.date.today()
#print(today)
startdate = today - datetime.timedelta(days=lookback)
enddate = today + datetime.timedelta(days=forecast)
#print(startdate)
#print(enddate)

BaseURL = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/'

ApiKey='YFPMR6XA4CTYQ26UNFK2C9Q2Y'

#UnitGroup sets the units of the output - us or metric
UnitGroup='us'

#Location for the weather data
Location1 = 'Banner%20Elk,NC'
Location2 = 'Valle%20Crucis,NC'
Location3 = 'Bethel,NC'

#Optional start and end dates
#If nothing is specified, the forecast is retrieved.
#If start date only is specified, a single historical or forecast day will be retrieved
#If both start and and end date are specified, a date range will be retrieved
StartDate = str(startdate)
EndDate = str(enddate)

#CSV format requires an 'include' parameter below to indicate which table section is required
ContentType="csv"

#specify elements to get
Elements='datetime%2Cprecip'

#include sections
#values include days,hours,current,alerts
Include="days"

#basic query including location
ApiQueryBE=BaseURL + Location1
ApiQueryVC=BaseURL + Location2
ApiQueryBTH=BaseURL + Location3

#append the start and end date if present
if (len(StartDate)):
    ApiQueryBE += "/" + StartDate
    ApiQueryVC += "/" + StartDate
    ApiQueryBTH += "/" + StartDate
    if (len(EndDate)):
        ApiQueryBE += "/" + EndDate
        ApiQueryVC += "/" + EndDate
        ApiQueryBTH += "/" + EndDate

#Url is completed. Now add query parameters (could be passed as GET or POST)
ApiQueryBE += "?"
ApiQueryVC += "?"
ApiQueryBTH += "?"

#append each parameter as necessary
if (len(UnitGroup)):
    ApiQueryBE += "&unitGroup=" + UnitGroup
    ApiQueryVC += "&unitGroup=" + UnitGroup
    ApiQueryBTH += "&unitGroup=" + UnitGroup

if (len(Elements)):
    ApiQueryBE += "&elements=" + Elements
    ApiQueryVC += "&elements=" + Elements
    ApiQueryBTH += "&elements=" + Elements

if (len(ContentType)):
    ApiQueryBE += "&contentType=" + ContentType
    ApiQueryVC += "&contentType=" + ContentType
    ApiQueryBTH += "&contentType=" + ContentType

if (len(Include)):
    ApiQueryBE += "&include=" + Include
    ApiQueryVC += "&include=" + Include
    ApiQueryBTH += "&include=" + Include

ApiQueryBE += "&key=" + ApiKey
ApiQueryVC += "&key=" + ApiKey
ApiQueryBTH += "&key=" + ApiKey

#query for Banner Elk
print(' - Running query URL: ', ApiQueryBE)
print()

try:
    CSVBytesBE = urllib.request.urlopen(ApiQueryBE)
    be = pd.read_table(CSVBytesBE, sep=",")
except urllib.error.HTTPError  as e:
    ErrorInfo= e.read().decode()
    print('Error code: ', e.code, ErrorInfo)
    sys.exit()
except  urllib.error.URLError as e:
    ErrorInfo= e.read().decode()
    print('Error code: ', e.code,ErrorInfo)
    sys.exit()

#query for Valle Crucis
print(' - Running query URL: ', ApiQueryVC)
print()

try:
    CSVBytesVC = urllib.request.urlopen(ApiQueryVC)
    vc = pd.read_table(CSVBytesVC, sep=",")
except urllib.error.HTTPError  as e:
    ErrorInfo= e.read().decode()
    print('Error code: ', e.code, ErrorInfo)
    sys.exit()
except  urllib.error.URLError as e:
    ErrorInfo= e.read().decode()
    print('Error code: ', e.code,ErrorInfo)
    sys.exit()

#query for Bethel
print(' - Running query URL: ', ApiQueryBTH)
print()

try:
    CSVBytesBTH = urllib.request.urlopen(ApiQueryBTH)
    bth = pd.read_table(CSVBytesBTH, sep=",")
except urllib.error.HTTPError  as e:
    ErrorInfo= e.read().decode()
    print('Error code: ', e.code, ErrorInfo)
    sys.exit()
except  urllib.error.URLError as e:
    ErrorInfo= e.read().decode()
    print('Error code: ', e.code,ErrorInfo)
    sys.exit()

#create date list for index
dates = pd.to_datetime(be["datetime"], format="%Y-%m-%d")

# combine multiple dataframes into one with readable columns
og = pd.concat([be["precip"], vc["precip"], bth["precip"]], axis=1)
og.index = dates
og.columns = ["be_precip", "vc_precip", "bth_precip"]

# helper for manipulating data into model readable format
def convert2matrix(data, look_back):
    X = []
    for i in range(len(data) - look_back):
        d = i + look_back
        X.append(data.iloc[i:d+1, 0:3])
    return np.array(X)

#print(og)

testvals = convert2matrix(og, lookback)

#load saved model
daModel = keras.models.load_model('taugModel')

prediction = daModel.predict(testvals)


def prediction_plot(test_predict):
    fig, ax = plt.subplots(2, figsize=(8, 8))
    ax[0].plot(og.index[lookback:], test_predict, 'r', label="prediction")
    ax[1].plot(og.index[lookback:], og.iloc[lookback:, 0], 'b', label="be_precip", alpha=0.3)
    ax[1].plot(og.index[lookback:], og.iloc[lookback:, 1], 'b', label="vc_precip", alpha=0.3)
    ax[1].plot(og.index[lookback:], og.iloc[lookback:, 2], 'b', label="bth_precip", alpha=0.3)
    ax[0].set_ylabel('Level (cfs)', size=15)
    ax[1].set_ylabel('Expected Rainfall Amount', size=15)
    ax[0].set_xlabel('Date', size=15)
    ax[1].set_xlabel('Date', size=15)
    ax[0].legend(fontsize=15)
    ax[1].legend(fontsize=15)
    plt.tight_layout()
    plt.show()


prediction_plot(prediction)