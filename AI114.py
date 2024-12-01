import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, BatchNormalization, Activation, Input
from tensorflow.keras.optimizers import Adam

# Load and preprocess CIFAR-10 data
(x_train, _), (_, _) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0

# Define dimensions and optimizer
noise_dim = 100
adam = Adam(lr=0.0002, beta_1=0.5)

# Generator Model
def build_generator():
    model = Sequential()
    model.add(Dense(256 * 8 * 8, activation='relu', input_dim=noise_dim))
    model.add(Reshape((8, 8, 256)))
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(3, (3, 3), padding='same'))
    model.add(Activation('tanh'))
    return model

# Discriminator Model
def build_discriminator():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=(32, 32, 3)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

# Build and compile the discriminator
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

# Build the generator
generator = build_generator()

# Create the GAN combining both models
z = Input(shape=(noise_dim,))
img = generator(z)
discriminator.trainable = False
validity = discriminator(img)

gan = Model(z, validity)
gan.compile(loss='binary_crossentropy', optimizer=adam)

# Training the GAN
def train_gan(epochs, batch_size=128, sample_interval=50):
    half_batch = batch_size // 2

    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, x_train.shape[0], half_batch)
        imgs = x_train[idx]

        noise = np.random.normal(0, 1, (half_batch, noise_dim))
        gen_imgs = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        valid_y = np.ones((batch_size, 1))

        g_loss = gan.train_on_batch(noise, valid_y)

        if epoch % sample_interval == 0:
            print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {d_loss[1]}] [G loss: {g_loss}]")

# Train the GAN
train_gan(epochs=10000, batch_size=64, sample_interval=1000)
