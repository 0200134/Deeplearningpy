import openai
import requests
from PIL import Image
from io import BytesIO
import os

# Set up your OpenAI API key
openai.api_key = 'YOUR_API_KEY_HERE'

# Function to generate image from text prompt using OpenAI's DALL-E model
def generate_images(prompt, n=1, size="512x512"):
    try:
        response = openai.Image.create(
            prompt=prompt,
            n=n,
            size=size
        )
        return response['data']
    except Exception as e:
        print(f"Error generating image: {e}")
        return []

# Function to save images locally
def save_images(images, directory='generated_images'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i, img_data in enumerate(images):
        image_url = img_data['url']
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        img_path = os.path.join(directory, f"image_{i+1}.png")
        img.save(img_path)
        print(f"Image saved at {img_path}")

# Example usage
if __name__ == "__main__":
    prompt = "A futuristic cityscape with flying cars and neon lights"
    images = generate_images(prompt, n=3)
    if images:
        save_images(images)
    else:
        print("No images generated.")
