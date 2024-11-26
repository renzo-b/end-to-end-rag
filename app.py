import os
import time

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true" # for langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

groq_api_key = os.getenv("GROQ_API_KEY")
openai_key = os.getenv("OPEN_API_KEY")

if "vector" not in st.session_state:
    st.session_state.embeddings = OpenAIEmbeddings(api_key=openai_key)
    st.session_state.loader = WebBaseLoader("https://en.wikipedia.org/wiki/Electric_vehicle")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(
        st.session_state.docs[:50])
    st.session_state.vectors = FAISS.from_documents(
        st.session_state.final_documents, st.session_state.embeddings)
    

st.title("Chat Demo With RAG")
llm = ChatGroq(groq_api_key=groq_api_key, model="Gemma-7b-It")

prompt_template = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
</context>
Question: {input}
"""
)

document_chain = create_stuff_documents_chain(llm, prompt_template)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)


prompt = st.text_input("Input your question")

if prompt:
    start = time.process_time()
    response = retrieval_chain.invoke({"input":prompt})
    print(f"Response time: {time.process_time()-start}")
    st.write(response["answer"])

    with st.expander("Document Similarity Search"):
    # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")


