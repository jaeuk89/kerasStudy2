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
x3 = np.array([i for i in range(1001,1101)])
y3 = np.array([i for i in range(1001,1101)])

print('x1.shape :', x1.shape)
print('type(x1) :', type(x1))
print(x1)

# train_set
x1_train = x1[:7]
y1_train = y1[:7]
x2_train = x2[:7]
y2_train = y2[:7]
x3_train = x3[:70]
y3_train = y3[:70]

# test_set
x1_test = x1[7:]
y1_test =y1[7:]
x2_test = x2[7:]
y2_test =y2[7:]
x3_test = x3[70:]
y3_test =y3[70:]

print('x1_train : ', x1_train)
print('x1_test : ', x1_test)
print('x1_train.shape :', x1_train.shape)
print('x1_test.shape :', x1_test.shape)

# 로드
from keras.models import model_from_json
json_file = open("model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights("model.h5")
print("Loaded model from disk")

# 모델 컴파일 후 Evaluation
model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['mse'])

loss1, acc1 = model.evaluate(x1_test, y1_test, batch_size=1)
loss2, acc2 = model.evaluate(x2_test, y2_test, batch_size=1)
loss3, acc3 = model.evaluate(x3_test, y3_test, batch_size=1)
predict1, predict2, predict3 = model.predict(x1_test), model.predict(x2_test), model.predict(x3_test)
print("loss1 : ", loss1,"acc1 : ", acc1,
      "\nloss2",loss2,"acc2",acc2,
      "\nloss3",loss3,"acc3",acc3,
      "\npredcit1 : ", predict1,
      "\npredict2",predict2,
      "\npredict3",predict3)

# R2 구하기
from sklearn.metrics import r2_score
r2_y1_predict = r2_score(y1_test, predict1)
r2_y2_predict = r2_score(y2_test, predict2)
r2_y3_predict = r2_score(y3_test, predict3)

print("R2_y1 : ", r2_y1_predict,
      " R2_y2 : ", r2_y2_predict,
      " R2_y3 : ", r2_y3_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE1 : ", RMSE(y1_test, predict1),
      " RMSE2 : ", RMSE(y2_test, predict2),
      " RMSE3 : ", RMSE(y3_test, predict3))