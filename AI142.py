import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, LeakyReLU, Dropout, Conv2D, MaxPooling2D, Flatten

# Define the generator model
def build_generator():
    model = Sequential()
    model.add(Dense(128, input_dim=100, activation='relu'))
    model.add(Dense(784, activation='sigmoid'))
    model.add(Reshape((28, 28, 1)))
    return model

# Define the discriminator model
def build_discriminator():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), strides=(2, 2), padding='same', input_shape=(28, 28, 1)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

# Create the generator and discriminator models
generator = build_generator()
discriminator = build_discriminator()

# Compile the discriminator
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Combine the generator and discriminator into a GAN model
discriminator.trainable = False
model = Sequential()
model.add(generator)
model.add(discriminator)
model.compile(loss='binary_crossentropy', optimizer='adam')

# Train the GAN
epochs = 100
batch_size = 128

for epoch in range(epochs):
    # Train the discriminator
    noise = np.random.randn(batch_size, 100)
    generated_images = generator.predict(noise)
    x = np.concatenate([generated_images, X_train[:batch_size]])
    y = np.concatenate([np.ones(batch_size), np.zeros(batch_size)])
    d_loss_real = discriminator.train_on_batch(X_train[:batch_size], y[:batch_size])
    d_loss_fake = discriminator.train_on_batch(generated_images, y[batch_size:])
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train the generator
    noise = np.random.randn(batch_size, 100)
    y = np.ones(batch_size)
    g_loss = model.train_on_batch(noise, y)

    print(f"Epoch {epoch+1}/{epochs}, D Loss: {d_loss}, G Loss: {g_loss}")
