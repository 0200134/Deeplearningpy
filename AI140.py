from transformers import pipeline

# Load a pre-trained Stable Diffusion model
pipe = pipeline("text-to-image")

# Generate an image from a text prompt
prompt = "A futuristic cyberpunk city with flying cars and neon lights"
image = pipe(prompt, guidance_scale=8.5)

# Display the generated image
image.show()
