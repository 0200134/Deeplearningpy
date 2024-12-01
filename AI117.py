import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import glue_convert_examples_to_features
from transformers import glue_processors as processors
import tensorflow_datasets as tfds

# Load the dataset
data, info = tfds.load('glue/sst2', with_info=True)
train_dataset = data['train']
validation_dataset = data['validation']

# Load BERT tokenizer and model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForSequenceClassification.from_pretrained(model_name)

# Preprocess the data
def encode(examples):
    return tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=128)

train_dataset = train_dataset.map(encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
validation_dataset = validation_dataset.map(encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_dataset = train_dataset.shuffle(10000).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
validation_dataset = validation_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])

# Train the model
model.fit(train_dataset, epochs=3, validation_data=validation_dataset)

# Evaluate the model
loss, accuracy = model.evaluate(validation_dataset)
print(f"Validation accuracy: {accuracy:.4f}")
