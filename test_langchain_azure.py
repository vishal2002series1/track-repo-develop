import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage

# Load the environment variables
env_path = find_dotenv()
load_dotenv(env_path, override=True)

def test_langchain_azure():
    print("🔄 Initializing LangChain AzureChatOpenAI...")
    
    subscription_key = os.getenv("API_KEYS")
    if not subscription_key:
        print("❌ ERROR: 'API_KEYS' not found in .env file.")
        return

    try:
        # Initialize the LangChain wrapper with your proven configuration
        llm = AzureChatOpenAI(
            api_key=subscription_key,
            azure_endpoint="https://wamaiexperiments.cognitiveservices.azure.com/",
            api_version="2024-12-01-preview",
            azure_deployment="gpt-5.4",
            temperature=0.0,
            max_tokens=1000
        )
        
        print("📤 Sending test prompt through LangChain to gpt-5.4...")
        
        # In LangChain, we pass a list of message objects
        messages = [
            HumanMessage(content="Hello! Are you online and ready to execute LangGraph tools?")
        ]
        
        response = llm.invoke(messages)
        
        print("\n✅ SUCCESS! LangChain connected to GPT. Response:")
        print("=" * 50)
        print(response.content)
        print("=" * 50)
        
    except Exception as e:
        print("\n❌ ERROR: LangChain failed to connect.")
        print(str(e))

if __name__ == "__main__":
    test_langchain_azure()