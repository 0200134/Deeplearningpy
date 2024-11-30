import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 모델 초기화
model = Sequential()

# 입력층
model.add(Dense(units=128, activation='relu', input_shape=(input_dim,)))

# 여러 개의 숨겨진 레이어 추가
for _ in range(200):  # 레이어 수는 필요에 따라 조정 가능
    model.add(Dense(units=128, activation='relu'))

# 출력층
model.add(Dense(units=num_classes, activation='softmax'))

# 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 요약 출력
model.summary()
