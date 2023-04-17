import pandas as pd
import numpy as np
from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt   
#%%
iris = datasets.load_iris()
df = pd.DataFrame(iris['data'], columns=iris['feature_names'])

#%%
'資料轉換-標準化'
min_max_scaler = preprocessing.MinMaxScaler()
df = pd.DataFrame(min_max_scaler.fit_transform(df),columns=iris['feature_names'])

#%%
labels = np.array(df['sepal length (cm)']) #應變數
features= df.drop('sepal length (cm)', axis = 1) #自變數

#%%
'隨機抽樣'
trainX, testX, trainY, testY = train_test_split(features, labels, test_size = 0.3, random_state = 42)
trainY = trainY.reshape(-1, 1) #創行
testY = testY.reshape(-1, 1) #創行
#%%
'建模'
model = Sequential()
model.add(Dense(64, input_dim=3, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1)) #類別型, activation='softmax'
model.compile(loss='mse', #類別型'categorical_crossentropy'
              optimizer=SGD(lr=0.1),#adam
              metrics=['mse','mape']) #類別型'accuracy'
#%%