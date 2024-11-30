import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# Create a Sequential model
model = Sequential()

# Add 200 layers with regularization and normalization
for _ in range(200):
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

# Output layer
model.add(Dense(10, activation='softmax'))

# Compile the model with an adaptive learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with early stopping and reduced learning rate on plateau
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

callbacks = [
    EarlyStopping(patience=10),
    ReduceLROnPlateau(factor=0.5, patience=5)
]

model.fit(X_train, y_train, epochs=100, batch_size=32, callbacks=callbacks)
