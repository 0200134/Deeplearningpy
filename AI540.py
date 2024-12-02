import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create a simple neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),  # Input layer
    Dense(64, activation='relu'),                     # Hidden layer
    Dense(1)                                          # Output layer
])

# Compile the model
model.compile(optimizer='adam',
              loss='mean_squared_error')

# Generate some example data
import numpy as np
x_train = np.random.rand(100, 10)  # 100 samples, 10 features each
y_train = np.random.rand(100, 1)   # 100 target values

# Train the model
model.fit(x_train, y_train, epochs=10)

# Make predictions
x_test = np.random.rand(10, 10)    # 10 new samples
predictions = model.predict(x_test)

print(predictions)
