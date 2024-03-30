import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain_community.document_loaders import TextLoader

# Load environment variables from .env file
load_dotenv()

def main():
    # Retrieve the OpenAI API key from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OpenAI API key not found. Please check your .env file.")
    
    # Initialize the OpenAI LLM with your API key and specify the model
    chat = OpenAI(api_key=api_key, model_name="gpt-3.5-turbo-instruct")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an assistant in creating all aspects of Crisis Management Exercises. Answer all questions to the best of your ability.                ",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    chain = prompt | chat
    
    Chat_History = ChatMessageHistory()
    Chat_History.add_user_message("Write a short crisis response scenario with 1 inject suitable for wildfire training for paramedics and firefighters.")
    
    response = chain.invoke({"messages":Chat_History.messages})
    Chat_History.add_ai_message(response);
    print(response) 
    
    Chat_History.add_user_message("Repeat your previous response in French!")
    
    response = chain.invoke({"messages": Chat_History.messages})
    
    print(response)
    

if __name__ == "__main__":
    main()
