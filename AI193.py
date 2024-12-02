import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import tensorflow_datasets as tfds
import os

# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Define an advanced ResNet model
def build_resnet_model(num_classes):
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        input_shape=(224, 224, 3),
        weights='imagenet'
    )
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax', dtype='float32')(x)  # specify dtype for mixed precision

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Load and preprocess the dataset
def preprocess_data(image, label):
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

(ds_train, ds_test), ds_info = tfds.load(
    'mnist', split=['train', 'test'], shuffle_files=True, as_supervised=True, with_info=True
)
ds_train = ds_train.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE).batch(32).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE).batch(32).prefetch(tf.data.AUTOTUNE)

# Initialize and compile the model
num_classes = ds_info.features['label'].num_classes
model = build_resnet_model(num_classes)
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

# Custom training loop with GradientTape
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    t_loss = loss_fn(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# Training the model with custom loop
EPOCHS = 20

for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in ds_train:
        train_step(images, labels)

    for test_images, test_labels in ds_test:
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))

# Save the model
model.save('advanced_resnet_model.h5')

# Load the model
loaded_model = tf.keras.models.load_model('advanced_resnet_model.h5')

# Evaluate the model
loaded_model.evaluate(ds_test)
