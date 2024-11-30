from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('your_model.h5')

# Define a route to handle image classification requests
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request
    image = request.files['image'].read()

    # Preprocess the image (e.g., resize, normalize)
    # ...

    # Make a prediction
    prediction = model.predict(preprocessed_image)

    # Return the prediction as JSON
    return jsonify({'prediction': str(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
