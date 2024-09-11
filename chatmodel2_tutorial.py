from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI


model=ChatGoogleGenerativeAI(model="gemini-1.5-pro")
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(
        content="You are a helpful assistant! Your name is Bob."#context for the conversation
    ),
    HumanMessage(
        content="What is your name?"
    )
]

# Define a chat model and invoke it with the messages
result=model.invoke(messages)

print(result.content)