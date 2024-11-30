import torch

# Load the YOLOv8 model
model = torch.hub.load('ultralytics/yolov8', 'yolov8n')  # Choose a model size (n, s, m, l, x)

# Inference on an image
results = model('path/to/image.jpg')  # Replace with your image path

# Display the results
results.print()
results.show()
results.save()
# Import necessary libraries
import torch

# Load a dataset (e.g., COCO)
dataset = torch.load('path/to/dataset.pt')

# Create a YOLOv8 model
model = torch.hub.load('ultralytics/yolov8', 'yolov8n').to('cuda')

# Train the model
model.train(dataset=dataset, epochs=10, imgsz=640, batch_size=16)
