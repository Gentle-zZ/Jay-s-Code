import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense


df_szsb = pd.read_csv('C:/Users/Jay/Documents/Code/Project/MATH620165/data/Stock/深证B股指数历史数据.csv')
df_200020 = pd.read_csv('C:/Users/Jay/Documents/Code/Project/MATH620165/data/Stock/200020历史数据.csv')
df_200045 = pd.read_csv('C:/Users/Jay/Documents/Code/Project/MATH620165/data/Stock/200045历史数据.csv')
df_200625 = pd.read_csv('C:/Users/Jay/Documents/Code/Project/MATH620165/data/Stock/200625历史数据.csv')
df_200725 = pd.read_csv('C:/Users/Jay/Documents/Code/Project/MATH620165/data/Stock/200725历史数据.csv')
df_200992 = pd.read_csv('C:/Users/Jay/Documents/Code/Project/MATH620165/data/Stock/200992历史数据.csv')

data_init = pd.DataFrame()
data_init['szsb'] = df_szsb.收盘
data_init['200020'] = df_200020.收盘
data_init['200045'] = df_200045.收盘
data_init['200625'] = df_200625.收盘
data_init['200725'] = df_200725.收盘
data_init['200992'] = df_200992.收盘


data_init.szsb = data_init.szsb.apply(lambda x:x.replace(',',''))
data_init.szsb = list(map(float,data_init.szsb))
data_init = data_init.dropna()


X = np.asarray(data_init.iloc[:,1:6])
y = np.asarray(data_init.iloc[:,0])

X_train, X_test, y_train, y_test = train_test_split(X,y)

# mean = X_train.mean(axis=0)
# std = X_train.std(axis=0)
# X_train -= mean
# X_train /= std
#
# X_test -= mean
# X_test -= std

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1], )))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mae'])
model.fit(X_train,y_train,epochs=400,batch_size=16,verbose=0)

mse_score,mae_score = model.evaluate(X_test,y_test)
print("MSE score:",format(mse_score))
print("MAE score:",format(mae_score))