import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Configure Google Gemini API
genai.configure(api_key=api_key)

# # Define the summarizer function
# def summarize_text(input_text):
#     # Define the prompt for summarization
#     prompt = (
#         "Please provide a concise summary of the following text. "
#         "Keep the summary brief and to the point:\n\n"
#         f"{input_text}\n\n"
#         "Summary:"
#     )
    
#     # Generate summary using Google Gemini API
#     response = genai.chat(
#         model="models/embedding-001",
#         prompt=prompt,
#         temperature=0.3
#     )
    
#     return response['text']

# # Streamlit UI
# st.title("📝 Google Gemini Text Summarizer")
# st.write("Enter some text and get a summary!")

# # Text input
# input_text = st.text_area("Input Text", height=250)

# # Summarize Button
# if st.button("Summarize"):
#     if input_text.strip():
#         with st.spinner("Summarizing..."):
#             summary = summarize_text(input_text)
#             st.subheader("Summary:")
#             st.write(summary)
#     else:
#         st.warning("Please enter some text to summarize.")
