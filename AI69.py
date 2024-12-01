import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, create_optimizer

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2LMHeadModel.from_pretrained("gpt2")

# Sample dataset
data = [
    "The quick brown fox jumps over the lazy dog.",
    "AI is transforming the way we interact with technology.",
    "Natural language processing enables machines to understand human language.",
]

# Encode the data
encoded_data = [tokenizer.encode(text, return_tensors='tf') for text in data]
input_ids = tf.concat(encoded_data, axis=1)


# Enable mixed precision training
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Optimizer with mixed precision
EPOCHS = 3
BATCH_SIZE = 1
LEARNING_RATE = 1e-5

optimizer, _ = create_optimizer(init_lr=LEARNING_RATE, num_train_steps=len(input_ids[0])*EPOCHS, num_warmup_steps=0)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    for i in range(0, len(input_ids[0]), BATCH_SIZE):
        input_batch = input_ids[:, i:i+BATCH_SIZE]
        with tf.GradientTape() as tape:
            logits = model(input_batch, training=True).logits
            loss = loss_fn(input_batch[:, 1:], logits[:, :-1, :])
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print(f"Batch {i // BATCH_SIZE + 1}/{len(input_ids[0]) // BATCH_SIZE}, Loss: {loss.numpy()}")
        
        
       
def generate_text(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors='tf')
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example usage
prompt = "In a future where technology"
generated_text = generate_text(prompt)
print(generated_text) 