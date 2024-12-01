import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, Reshape, Flatten, Dropout
from tensorflow.keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt

# Set up directories for generated images
os.makedirs("generated_images", exist_ok=True)

# Parameters
latent_dim = 100  # Size of the random noise vector
epochs = 10000
batch_size = 64
image_save_interval = 500  # Save generated images every X epochs

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

# Create models
generator = build_generator(latent_dim)
discriminator = build_discriminator()

# Compile discriminator
discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), 
                      loss='binary_crossentropy', metrics=['accuracy'])

# Freeze discriminator weights during generator training
discriminator.trainable = False
gan_input = tf.keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), loss='binary_crossentropy')

# Function to save generated images
def save_generated_images(epoch, generator, latent_dim):
    noise = np.random.normal(0, 1, (16, latent_dim))
    generated_images = generator.predict(noise)
    generated_images = (generated_images + 1) / 2.0  # Rescale from [-1, 1] to [0, 1]
    
    plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(generated_images[i], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"generated_images/epoch_{epoch}.png")
    plt.close()

# Training GAN
def train_gan(generator, discriminator, gan, epochs, batch_size, latent_dim):
    (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train.astype('float32') - 127.5) / 127.5  # Normalize to [-1, 1]
    x_train = np.expand_dims(x_train, axis=-1)

    batch_count = x_train.shape[0] // batch_size

    for epoch in range(epochs):
        for _ in range(batch_count):
            # Generate noise and create fake images
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            fake_images = generator.predict(noise)
            
            # Select a random batch of real images
            real_images = x_train[np.random.randint(0, x_train.shape[0], batch_size)]

            # Train discriminator
            x_combined = np.concatenate([real_images, fake_images])
            y_combined = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
            d_loss = discriminator.train_on_batch(x_combined, y_combined)

            # Train generator
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            y_gan = np.ones((batch_size, 1))  # Labels for generated images
            g_loss = gan.train_on_batch(noise, y_gan)

        # Print losses
        print(f"Epoch {epoch + 1}/{epochs} | D Loss: {d_loss[0]:.4f}, D Accuracy: {d_loss[1]:.4f}, G Loss: {g_loss:.4f}")

        # Save generated images at intervals
        if (epoch + 1) % image_save_interval == 0:
            save_generated_images(epoch + 1, generator, latent_dim)

# Start training
train_gan(generator, discriminator, gan, epochs, batch_size, latent_dim)
