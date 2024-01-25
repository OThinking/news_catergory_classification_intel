import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

X_train, X_test, Y_train, Y_test = np.load('./news_data_max_20_wordsize_10854.npy', allow_pickle=True)
# allow_pickle=True: 피클배열을 읽을 수 있다.
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

model = Sequential()
model.add(Embedding(10854, 300, input_length=20)) # 자연어를 학습하는 layer / 10855개의 차원공간을 만들고 300개(느낌적인 감으로 설정)의 차원공간으로 줄인다
# 차원공간이 많으면 많을 수록 밀도는 작아진다 즉 각 요소들간의 백터적인 거리가 늘어난다 이 의미는 데이터가 희소해진다 => 차원의 저주라고도 한다
# 어느 방향의 축을 만들어 그 축으로 투사를 하여 차원을 줄이는 방법을 차원 축소라고 한다 / 대신 데이터간의 관계를 최대한 배율을 유지하면서 만든다
model.add(Conv1D(32, kernel_size=5, padding='same', activation='relu')) # 주변에 있는 위치 관계를 학습하기 위해 Conv1D를 사용한다
model.add(MaxPooling1D(pool_size=1)) # pool_size가 1이기 때문에 아무일을 하지 않지만 Conv1D 와 세트로 쓰기도 하고 실험을 통해 최적을 찾기위해 넣어놓기는 했다
model.add(LSTM(128, activation='tanh', return_sequences=True)) # return_sequences: 입력이 하나씩 들어갈때마다 나온 결과값을 쭉 연결시킨다
# return_sequences를 사용하지 않으면 제일 마지막 하나만 들어가기에 뒤에 LSTM이 있을때는 사용해줘야한다.
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
fit_hist = model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_test, Y_test))
model.save('./models/news_category_classification_model_{}.h5'.format(fit_hist.history['val_accuracy'][-1]))
plt.plot(fit_hist.history['val_accuracy'], label='validation accuracy')
plt.plot(fit_hist.history['accuracy'], label='train accuracy')
plt.legend()
plt.show()


