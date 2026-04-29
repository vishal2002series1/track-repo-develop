import os
from dotenv import load_dotenv
from openai import AzureOpenAI
 
load_dotenv()
subscription_key = os.getenv("API_KEYS")
# client = AzureOpenAI( api_version="2024-12-01-preview", azure_endpoint="https://wamaiexperiments.cognitiveservices.azure.com/", api_key=subscription_key )
endpoint = "https://wamaiexperiments.cognitiveservices.azure.com/"
model_name = "gpt-5.4"
deployment = "gpt-5.4"
api_version = "2024-12-01-preview"
client = AzureOpenAI( api_version=api_version, azure_endpoint=endpoint, api_key=subscription_key)
response = client.chat.completions.create( messages=[ { "role": "system", "content": "You are a helpful assistant.", }, { "role": "user", "content": "I am going to Paris, what should I see?", } ], max_completion_tokens=16384, model=deployment )
print(response.choices[0].message.content)