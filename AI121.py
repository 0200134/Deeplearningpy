import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, Reshape, Flatten, Dropout
from tensorflow.keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt

# Define input shape
latent_dim = 100  # Dimension of the random noise

# Generator Model
def build_generator(latent_dim):
    model = Sequential([
        Dense(256, input_dim=latent_dim),
        LeakyReLU(alpha=0.2),
        Dense(512),
        LeakyReLU(alpha=0.2),
        Dense(1024),
        LeakyReLU(alpha=0.2),
        Dense(28 * 28, activation='tanh'),
        Reshape((28, 28))
    ])
    return model

# Discriminator Model
def build_discriminator(input_shape=(28, 28)):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(512),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        Dense(256),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    return model

# Build the models
generator = build_generator(latent_dim)
discriminator = build_discriminator()

# Compile discriminator
discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), 
                      loss='binary_crossentropy', metrics=['accuracy'])

# Combined GAN model
discriminator.trainable = False  # Freeze discriminator during generator training
gan_input = tf.keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), loss='binary_crossentropy')

# Training GAN
def train_gan(generator, discriminator, gan, epochs=10000, batch_size=64):
    (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train.astype('float32') - 127.5) / 127.5  # Normalize to [-1, 1]
    x_train = np.expand_dims(x_train, axis=-1)
    batch_count = x_train.shape[0] // batch_size

    for epoch in range(epochs):
        for _ in range(batch_count):
            # Train discriminator
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            generated_images = generator.predict(noise)
            real_images = x_train[np.random.randint(0, x_train.shape[0], batch_size)]
            
            x_combined = np.concatenate([real_images, generated_images])
            y_combined = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
            
            d_loss = discriminator.train_on_batch(x_combined, y_combined)

            # Train generator
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            y_gan = np.ones((batch_size, 1))
            g_loss = gan.train_on_batch(noise, y_gan)

        # Print loss at intervals
        if epoch % 100 == 0:
            print(f"Epoch {epoch} / D Loss: {d_loss[0]} / G Loss: {g_loss}")

# Train the GAN
train_gan(generator, discriminator, gan)

# Generate and display an image
def generate_image(generator, latent_dim):
    noise = np.random.normal(0, 1, (1, latent_dim))
    generated_image = generator.predict(noise)[0]
    plt.imshow(generated_image, cmap='gray')
    plt.axis('off')
    plt.show()

generate_image(generator, latent_dim)
