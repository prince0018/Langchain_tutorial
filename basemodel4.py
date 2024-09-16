import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.tools.tavily_search import TavilySearchResults

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Configure Google Gemini API
genai.configure(api_key=google_api_key)

# Set up the LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# PDF Processing Functions
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details.
    If the answer is not in provided context just say, 'answer is not available in the context', don't provide wrong information.

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# Chatbot Functionality
def handle_chat():
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Type your message here..."):
        client = ChatGoogleGenerativeAI(model="gemini-pro")
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Generate a response
        response = client.invoke(st.session_state["messages"])
        msg = response.content
        st.session_state["messages"].append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)

# Web Search Functionality
def web_search(query):
    search = TavilySearchResults(api_key=tavily_api_key, max_results=2)
    search_results = search.invoke(query)
    return search_results

# Main Application Logic
def main():
    st.sidebar.header("📄 PDF Processing & Chatbot")
    
    # Dropdown menu for user to select between "Chat with PDF", "Talk to Chatbot", and "Talk to Web"
    option = st.sidebar.selectbox("Choose your option", ("Chat with PDF", "Talk to Chatbot", "Talk to Web"))

    if option == "Chat with PDF":
        st.sidebar.write("Upload PDF files and chat with them.")
        
        # PDF Upload and Processing Section
        pdf_docs = st.sidebar.file_uploader("Upload PDF Files", accept_multiple_files=True)
        
        if st.sidebar.button("Process PDF"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.sidebar.success("PDFs processed successfully!")
            else:
                st.sidebar.warning("Please upload PDF files first.")

        # PDF Query Section
        st.header("Ask Questions about PDF")
        user_question = st.text_input("Ask a question related to the uploaded PDFs")

        if user_question:
            with st.spinner("Searching for an answer..."):
                answer = user_input(user_question)
                st.write(f"**Answer:** {answer}")

    elif option == "Talk to Chatbot":
        # General Chatbot Section
        st.header("Chat with Gemini AI")
        handle_chat()
    
    elif option == "Talk to Web":
        # Web Search Section
        st.header("Search the Web")
        query = st.text_input("Enter your search query")

        if query:
            with st.spinner("Searching the web..."):
                results = web_search(query)
                st.write("**Search Results:**")
                st.write(results)

if __name__ == "__main__":
    main()
