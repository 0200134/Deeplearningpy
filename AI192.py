import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
import tensorflow_hub as hub
import os

# Define an advanced convolutional neural network model
class AdvancedCNN(Model):
    def __init__(self):
        super(AdvancedCNN, self).__init__()
        self.conv1 = Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))
        self.pool1 = MaxPooling2D((2, 2))
        self.conv2 = Conv2D(64, (3, 3), activation='relu')
        self.pool2 = MaxPooling2D((2, 2))
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.dropout = Dropout(0.5)
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.dropout(x)
        return self.d2(x)

# Load and preprocess the dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape(-1, 28, 28, 1) / 255.0
test_images = test_images.reshape(-1, 28, 28, 1) / 255.0

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    fill_mode='nearest'
)
datagen.fit(train_images)

# Initialize the model and compile
model = AdvancedCNN()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
learning_rate_schedule = LearningRateScheduler(lambda epoch: 1e-3 * 10 ** (-epoch / 20))
checkpoint_path = "training/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
model_checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                   save_weights_only=True,
                                   verbose=1)

# Training the model with advanced features
history = model.fit(datagen.flow(train_images, train_labels, batch_size=32),
                    epochs=50,
                    validation_data=(test_images, test_labels),
                    callbacks=[early_stopping, learning_rate_schedule, model_checkpoint])

# Save the entire model to a HDF5 file
model.save('advanced_cnn_model.h5')

# Load the model
new_model = tf.keras.models.load_model('advanced_cnn_model.h5')

# Fine-tuning a pre-trained model from TensorFlow Hub
hub_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"
feature_extractor_layer = hub.KerasLayer(hub_url, input_shape=(224, 224, 3), trainable=False)

# Building a model with the pre-trained feature extractor
fine_tune_model = Sequential([
    feature_extractor_layer,
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

fine_tune_model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

# Assume we have train_images_resized and test_images_resized resized to (224, 224)
# For illustration purposes, we will resize train_images and test_images here
train_images_resized = tf.image.resize(train_images, [224, 224])
test_images_resized = tf.image.resize(test_images, [224, 224])

# Training the fine-tuned model
fine_tune_model.fit(train_images_resized, train_labels, epochs=10, validation_data=(test_images_resized, test_labels))

# Evaluate the fine-tuned model
test_loss, test_acc = fine_tune_model.evaluate(test_images_resized, test_labels, verbose=2)
print('\nFine-tuned model test accuracy:', test_acc)
