import streamlit as st
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pdfplumber
import pandas as pd
import numpy as np

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to extract tables from PDFs
def get_pdf_tables(pdf_docs):
    tables = []
    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_reader:
            for page in pdf_reader.pages:
                page_tables = page.extract_tables()
                for table in page_tables:
                    df = pd.DataFrame(table[1:], columns=table[0])
                    tables.append(df)
    return tables

# Function to handle text chunks for Vector Store
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to get Vector Store using Chroma
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma.from_texts(text_chunks, embedding=embeddings, persist_directory="chroma_db")
    return vector_store

# Function to initialize conversational chain
def get_conversational_chain(vector_store):
    prompt = """
You are an AI assistant designed to summarize and extract information from documents. Follow these instructions carefully:

1. Summarization Task:
   - Summarize the provided document in no more than 200 words.
   - Ensure the summary is concise and captures the main points.

2. Formatting Requirements:
   - Use clear headings and bullet points to organize the information when needed.
   - If a table format is suitable for presenting the information, format the response accordingly.

3. Content Limitation:
   - Base your response strictly on the content of the provided document.
   - Do not use any external knowledge or information not contained in the document.

4. Use of Chat History:
   - Incorporate the conversation history to provide contextually relevant and coherent answers. Ensure that the response aligns with the previous discussion.

Example Input:

[Insert document text here]

Example Output:

Summary of the Document

- Main Point 1: Brief description of the main point.
- Main Point 2: Brief description of another key point.

Details:

- Section Title:
  - Bullet point with relevant detail.
  - Bullet point with additional information.

Table Example (if applicable):

| Column 1 | Column 2 |
|----------|----------|
| Detail 1 | Detail 2 |
| Detail 3 | Detail 4 |

Please ensure your response adheres to these instructions and formatting guidelines.
Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
------
<ctx>
{context}
</ctx>
------
<hs>
{history}
</hs>
------
{question}

Answer:
"""


    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt_template = PromptTemplate(
        template=prompt,
        input_variables=["context", "question", "history"]  # Added 'history' to match the memory input
    )

    memory = st.session_state.get("memory", ConversationBufferMemory(
        memory_key="history",
        input_key="question"
    ))

    if vector_store is None:
        st.error("Vector Store is not initialized. Please upload and process your documents first.")
        return None

    retrieval_chain = RetrievalQA.from_chain_type(llm=model,
                                                  chain_type='stuff',
                                                  retriever=vector_store.as_retriever(),
                                                  chain_type_kwargs={
                                                      "prompt": prompt_template,
                                                      "memory": memory
                                                  })

    st.session_state.memory = memory  # Persist memory in session state

    return retrieval_chain

# Function to handle user input and response
def user_input(user_question, vector_store):
    chain = get_conversational_chain(vector_store)
    
    if chain is None:
        st.write("No valid conversational chain found.")
        return

    response = chain(
        {"query": user_question}, return_only_outputs=True
    )

    if "result" in response:
        st.write("Reply: ", response["result"])
    else:
        st.write("No valid response received.")

# Main function to run Streamlit app
def main():
    st.set_page_config(page_title="Chat with Documents using Gemini")
    st.header("QA with Your Multiple Documents 💁")

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    
    if "response" not in st.session_state:
        st.session_state.response = ""

    user_question = st.text_input("Ask a Question from the PDF Files", value=st.session_state.response)

    if st.button("Clear Response"):
       st.session_state.user_question = ""  # Clear the input cell
       st.session_state.response = ""  # Clear the output cell
       st.session_state.memory = ConversationBufferMemory(
           memory_key="history",
           input_key="question"
       )  # Clear the memory

    if user_question:
        st.session_state.response = user_question
        user_input(user_question, st.session_state.vector_store)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                st.session_state.vector_store = get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
