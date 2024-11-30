from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

app = Flask(__name__)

# Load the pre-trained model
model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
model.eval()

# Define a route to handle image classification requests
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0)

    # Make a prediction
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output.data, 1)

    # Return the prediction as JSON
    return jsonify({'prediction': str(predicted.item())})

if __name__ == '__main__':
    app.run(debug=True)
