import tensorflow as tf
from tensorflow.keras import datasets, layers, models, applications, optimizers
import matplotlib.pyplot as plt

# Load and preprocess the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Build the EfficientNet model
def build_efficientnet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = data_augmentation(inputs)
    base_model = applications.EfficientNetB0(include_top=False, input_tensor=x, weights='imagenet')
    base_model.trainable = False  # Freeze the base model

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

# Build and compile the model
model = build_efficientnet((32, 32, 3), 10)
model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Unfreeze some layers of the base model for fine-tuning
fine_tune_at = 100
for layer in model.layers[:fine_tune_at]:
    layer.trainable = False
for layer in model.layers[fine_tune_at:]:
    layer.trainable = True

# Train the model
history = model.fit(train_images, train_labels, epochs=30, 
                    validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
