from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI


model=ChatGoogleGenerativeAI(model="gemini-1.5-pro")

chat_history=[]

system_message=SystemMessage(content="You are a helpful AI assistant.")
chat_history.append(system_message)

while True:
    query=input("You: ")
    if query.lower()=="exit":
        break
    chat_history.append(HumanMessage(content=query))

    result=model.invoke(chat_history)
    response=result.content
    chat_history.append(AIMessage(content=response))
    print(f"AI: {response}")

print("----Message History----")
print(chat_history)
#study about basechatmessagehistory in langchain
#you can also save chat history, look the video in langchain tutorial