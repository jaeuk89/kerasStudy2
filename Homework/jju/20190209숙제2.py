import numpy as np
import pandas as pd
from keras.layers import Dense,Activation
from keras.models import Sequential

# 변수 설정
epochs=11
batch_size=1

x1 = np.array([i for i in range(1,11)])
y1 = np.array([i for i in range(1,11)])

x2 = np.array([i for i in range(101,111)])
y2 = np.array([i for i in range(101,111)])

x = np.concatenate((x1,x2))
y = np.concatenate((y1,y2))

print(x.shape)
print(type(x))

x_train = np.concatenate((x[:7],x[10:17]))
y_train = np.concatenate((y[:7],y[10:17]))
x_test = np.concatenate((x[7:10],x[17:]))
y_test = np.concatenate((y[7:10],y[17:]))


print(x_train)
print(x_test)

# 모델 구성

model = Sequential()
model.add(Dense(10, input_shape=(1,), activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model.compile(loss='mse', optimizer='adam', metrics=['acc'])

model.summary()

model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))

a, b = model.evaluate(x_test, y_test, batch_size=1)
print(a, b)

y_predict = model.predict(x_test)
print(y_predict)


# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))
