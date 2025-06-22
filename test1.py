import ollama

# List models
print(ollama.list())

# Generate text
response = ollama.generate(model="llama2", prompt="Why is the sky blue?")
print(response['response'])