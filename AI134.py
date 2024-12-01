from transformers import pipeline

# Load a pre-trained BERT model for sentiment analysis
classifier = pipeline("sentiment-analysis")

# Example text
text = "This is a really great product! I love it."

# Perform sentiment analysis
result = classifier(text)

print(result)
