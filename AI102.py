import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, add
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# Load the InceptionV3 model pre-trained on ImageNet
image_model = InceptionV3(weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-2].output
image_features_extract_model = Model(inputs=new_input, outputs=hidden_layer)

# Load the tokenizer and mapping
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_length = 34  # Maximum length of a caption

# Define the model
def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Function to extract features from an image
def extract_features(filename):
    img = image.load_img(filename, target_size=(299, 299))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    features = image_features_extract_model.predict(img, verbose=0)
    return features

# Function to generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word[yhat]
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# Load image features
photo = extract_features('example.jpg')

# Generate description
model = define_model(vocab_size=len(tokenizer.word_index) + 1, max_length=max_length)
model.load_weights('model_weights.h5')
description = generate_desc(model, tokenizer, photo, max_length)
print(description)
