import openai

openai.api_key = "your-openai-key"

def get_embedding(text, model="text-embedding-3-small"):
    response = openai.Embedding.create(input=[text], model=model)
    return response['data'][0]['embedding']
