import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import coremltools as ct

# Define a sequential model
model = Sequential()

# Adding 100 dense layers
for _ in range(100):
    model.add(Dense(64, activation='relu'))

# Adding the output layer
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Assuming you have training data (x_train, y_train)
# model.fit(x_train, y_train, epochs=10)

# Convert the model to CoreML format
coreml_model = ct.convert(model)

# Save the model
coreml_model.save('DeepLearningModel.mlmodel')


