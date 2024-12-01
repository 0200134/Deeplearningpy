import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Custom weight scaling layer
class WeightScaling(tf.keras.layers.Layer):
    def __init__(self, gain, **kwargs):
        super().__init__(**kwargs)
        self.gain = gain

    def build(self, input_shape):
        fan_in = np.prod(input_shape[1:])
        self.scale = self.gain / np.sqrt(fan_in)
        self.bias = self.add_weight(shape=(input_shape[-1],), initializer='zeros', trainable=True, name='bias')

    def call(self, inputs):
        return inputs * self.scale + self.bias

# Pixelwise normalization layer
class PixelwiseNormalization(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs / tf.math.sqrt(tf.reduce_mean(inputs**2, axis=-1, keepdims=True) + 1e-8)

# Equalized learning rate convolutional layer
class EqualizedConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, gain=np.sqrt(2), **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.gain = gain

    def build(self, input_shape):
        fan_in = np.prod(input_shape[1:])
        self.scale = self.gain / np.sqrt(fan_in)
        self.conv = layers.Conv2D(self.filters, self.kernel_size, padding='same')

    def call(self, inputs):
        return self.conv(inputs) * self.scale

# Custom dense layer with equalized learning rate
class EqualizedDense(tf.keras.layers.Layer):
    def __init__(self, units, gain=np.sqrt(2), **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.gain = gain

    def build(self, input_shape):
        fan_in = np.prod(input_shape[1:])
        self.scale = self.gain / np.sqrt(fan_in)
        self.dense = layers.Dense(self.units)

    def call(self, inputs):
        return self.dense(inputs) * self.scale

# Apply noise layer
class ApplyNoise(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.noise = self.add_weight(shape=(1, *input_shape[1:]), initializer='zeros', trainable=True, name='noise')

    def call(self, inputs):
        return inputs + tf.random.normal(tf.shape(inputs)) * self.noise


def make_generator_model():
    noise = layers.Input(shape=(512,))
    label = layers.Input(shape=(1,))
    
    # Mapping Network
    x = EqualizedDense(512)(noise)
    x = PixelwiseNormalization()(x)
    for _ in range(7):
        x = EqualizedDense(512)(x)
        x = PixelwiseNormalization()(x)
    
    # Constant Input Layer
    x = layers.Input(shape=(4, 4, 512))
    
    # Initial block
    y = EqualizedConv2D(512, (3, 3))(x)
    y = ApplyNoise()(y)
    y = layers.LeakyReLU()(y)
    y = PixelwiseNormalization()(y)
    y = EqualizedConv2D(512, (3, 3))(y)
    y = ApplyNoise()(y)
    y = layers.LeakyReLU()(y)
    y = PixelwiseNormalization()(y)
    
    # Upsample and blocks
    for filters in [256, 128, 64, 32, 16, 8, 4]:
        y = layers.UpSampling2D()(y)
        y = EqualizedConv2D(filters, (3, 3))(y)
        y = ApplyNoise()(y)
        y = layers.LeakyReLU()(y)
        y = PixelwiseNormalization()(y)
        y = EqualizedConv2D(filters, (3, 3))(y)
        y = ApplyNoise()(y)
        y = layers.LeakyReLU()(y)
        y = PixelwiseNormalization()(y)
    
    # Final RGB layer
    y = EqualizedConv2D(3, (1, 1), gain=1.0)(y)
    y = layers.Activation('tanh')(y)
    
    return models.Model([noise, label], y)



def make_discriminator_model():
    image = layers.Input(shape=(256, 256, 3))
    
    # Initial block
    x = EqualizedConv2D(32, (3, 3))(image)
    x = layers.LeakyReLU()(x)
    
    # Downsampling blocks
    for filters in [32, 64, 128, 256, 512, 512, 512, 512]:
        x = EqualizedConv2D(filters, (3, 3))(x)
        x = layers.LeakyReLU()(x)
        x = layers.AveragePooling2D()(x)
    
    x = layers.Flatten()(x)
    x = EqualizedDense(512)(x)
    x = layers.LeakyReLU()(x)
    x = EqualizedDense(1)(x)
    
    return models.Model(image, x)


# Loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.0, beta_2=0.99)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.0, beta_2=0.99)

# Create the models
generator = make_generator_model()
discriminator = make_discriminator_model()

# Checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# Training loop
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    
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

        # Save the model every 10 epochs
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

# Start the training
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
seed = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, NOISE_DIM])
train(train_dataset, EPOCHS)
