import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")#importing the google api key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))#using the apikey

# Set the page configuration
st.set_page_config(page_title="PDF & Chatbot", layout="wide")
st.title("💬 PDF Chatbot with Gemini AI")


######################


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

# Main Application Logic
def main():
    st.sidebar.header("📄 PDF Processing & Chatbot")
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
    
    # General Chatbot Section
    st.header("Chat with Gemini AI")
    handle_chat()

if __name__ == "__main__":
    main()
