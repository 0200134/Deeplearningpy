import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt

# Spectral normalization layer
class SpectralNormalization(tf.keras.layers.Layer):
    def __init__(self, layer, **kwargs):
        super().__init__(**kwargs)
        self.layer = layer

    def build(self, input_shape):
        self.layer.build(input_shape)
        self.u = self.add_weight(shape=(1, self.layer.kernel.shape[-1]),
                                 initializer='random_normal',
                                 trainable=False, name='u')

    def call(self, inputs):
        w = self.layer.kernel
        w_shape = w.shape.as_list()
        w = tf.reshape(w, [-1, w_shape[-1]])

        u_hat = self.u
        v_hat = tf.linalg.matvec(w, u_hat, transpose_a=True)
        v_hat /= tf.norm(v_hat, ord=2)
        u_hat = tf.linalg.matvec(w, v_hat)
        u_hat /= tf.norm(u_hat, ord=2)

        sigma = tf.reduce_sum(tf.linalg.matvec(w, v_hat) * u_hat)
        self.u.assign(u_hat)

        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)
        self.layer.kernel = w_norm

        return self.layer(inputs)

# Self-Attention layer
class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.f = layers.Conv2D(input_shape[-1] // 8, kernel_size=1)
        self.g = layers.Conv2D(input_shape[-1] // 8, kernel_size=1)
        self.h = layers.Conv2D(input_shape[-1] // 2, kernel_size=1)
        self.v = layers.Conv2D(input_shape[-1], kernel_size=1)

    def call(self, inputs):
        f = self.f(inputs)
        g = self.g(inputs)
        h = self.h(inputs)

        s = tf.linalg.matmul(f, g, transpose_a=True)
        beta = tf.nn.softmax(s)

        o = tf.linalg.matmul(beta, h)
        o = self.v(o)

        return inputs + o
        
 def make_generator_model():
    noise = layers.Input(shape=(512,))
    label = layers.Input(shape=(1,))

    x = layers.Dense(4*4*512, use_bias=False)(noise)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((4, 4, 512))(x)

    for filters in [512, 256, 128, 64, 32]:
        x = layers.UpSampling2D()(x)
        x = SpectralNormalization(layers.Conv2D(filters, kernel_size=3, padding='same', use_bias=False))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        if filters == 64:
            x = SelfAttention()(x)

    x = SpectralNormalization(layers.Conv2D(3, kernel_size=3, padding='same', use_bias=False))(x)
    x = layers.Activation('tanh')(x)

    return models.Model([noise, label], x)

def make_discriminator_model():
    image = layers.Input(shape=(128, 128, 3))

    for filters in [32, 64, 128, 256, 512]:
        if filters == 64:
            x = SelfAttention()(x)
        x = SpectralNormalization(layers.Conv2D(filters, kernel_size=3, strides=2, padding='same'))(image)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)
    
    x = layers.Flatten()(x)
    x = SpectralNormalization(layers.Dense(1))(x)

    return models.Model(image, x)    
  
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

generator_optimizer = optimizers.Adam(1e-4, beta_1=0.0, beta_2=0.99)
discriminator_optimizer = optimizers.Adam(1e-4, beta_1=0.0, beta_2=0.99)

generator = make_generator_model()
discriminator = make_discriminator_model()

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, 512])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch)
        
        generate_and_save_images(generator, epoch + 1, seed)

        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print(f'Epoch {epoch + 1} completed')

    generate_and_save_images(generator, epochs, seed)

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((predictions[i] + 1) / 2)
        plt.axis('off')

    plt.savefig(f'image_at_epoch_{epoch:04d}.png')
    plt.show()

# Train the model
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
seed = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, 512])
train(train_dataset, EPOCHS)