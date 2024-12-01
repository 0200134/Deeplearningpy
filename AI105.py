import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Reshape, Flatten, LeakyReLU
from tensorflow.keras.models import Sequential

# Load MNIST dataset
(x_train, _), (_, _) = mnist.load_data()
x_train = x_train / 255.0

# Reshape images to a single dimension
x_train = x_train.reshape(x_train.shape[0], 784)

# Define the generator model
generator = Sequential([
    Dense(128, input_dim=100),
    LeakyReLU(0.2),
    Dense(784, activation='sigmoid'),
    Reshape((28, 28))
])

# Define the discriminator model
discriminator = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the discriminator
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Combine the generator and discriminator
discriminator.trainable = False
model = Sequential([generator, discriminator])
model.compile(loss='binary_crossentropy', optimizer='adam')

# Train the GAN
batch_size = 256
epochs = 100

for epoch in range(epochs):
    # Train the discriminator
    noise = tf.random.normal([batch_size, 100])
    generated_images = generator.predict(noise)

    x = tf.concat([x_train[:batch_size], generated_images], axis=0)
    y = tf.concat([tf.ones([batch_size, 1]), tf.zeros([batch_size, 1])], axis=0)

    d_loss_real = discriminator.train_on_batch(x_train[:batch_size], y[:batch_size])
    d_loss_fake = discriminator.train_on_batch(generated_images, y[batch_size:])
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train the generator
    noise = tf.random.normal([batch_size, 100])
    y = tf.ones([batch_size, 1])
    g_loss = model.train_on_batch(noise, y)

    print(f"Epoch {epoch+1}/{epochs}, D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")

# Generate and save images
noise = tf.random.normal([16, 100])
generated_images = generator.predict(noise)
