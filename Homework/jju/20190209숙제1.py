from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM
from keras import layers
from keras import Input
import numpy as np
import pandas as pd
import keras
import tensorflow as tf

np.random.seed(777)

# 데이터 생성
x1 = np.array([i for i in range(1,11)])
y1 = np.array([i for i in range(1,11)])

x2 = np.array([i for i in range(101,111)])
y2 = np.array([i for i in range(101,111)])


print(x1,x2)
print(type(x1),type(x2))
print(x1.shape,x2.shape)

# 훈련과 검증 분리
x1_train, x1_test = x1[:7], x1[7:]
y1_train, y1_test = y1[:7], y1[7:]

x2_train, x2_test = x2[:7], x2[7:]
y2_train, y2_test = y2[:7], y2[7:]

print(x1_train,x2_train)
print(x1_train.shape,x2_train.shape)
print(x1_test.shape,x2_test.shape)


# 모델 구성하기

model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1, activation='relu'))


# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.summary()

model.fit(x1_train, y1_train, epochs= 10, batch_size=1, validation_data=(x1_test, y1_test))
model.fit(x2_train, y2_train, epochs= 10, batch_size=1, validation_data=(x2_test, y2_test))

loss1, acc1 = model.evaluate(x1_test, y1_test, batch_size=1)
loss2, acc2 = model.evaluate(x2_test, y2_test, batch_size=1)
predict1, predict2 = model.predict(x1_test),model.predict(x2_test)
print("loss1 : ", loss1,"acc1 : ", acc1,"\nloss2",loss2,"acc2",acc2,"\npredcit1 : ", predict1,"\npredict2",predict2)


# R2 구하기
from sklearn.metrics import r2_score
r2_y1_predict = r2_score(y1_test, predict1)
r2_y2_predict = r2_score(y2_test, predict2)
print("R2_y1 : ", r2_y1_predict," R2_y2 : ", r2_y2_predict,)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE1 : ", RMSE(y1_test, predict1)," RMSE2 : ", RMSE(y2_test, predict2))