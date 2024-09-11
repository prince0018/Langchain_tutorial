from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
model=ChatGoogleGenerativeAI(model="gemini-pro")

result = model.invoke("Write a poem about Milk way universe")
print(result.content)