import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import tempfile
import os

st.set_page_config(page_title="PDF Q&A with Ollama", layout="wide")
st.title("📄 Ask Questions from PDF using Ollama + HuggingFace Embeddings")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.success("PDF uploaded successfully!")

    # Load and split PDF
    st.info("Loading and processing PDF...")
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    if not chunks:
        st.error("No content found in PDF after processing.")
        st.stop()

    # HuggingFace Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(chunks, embedding=embeddings)

    # Ollama LLM
    llm = ChatOllama(model="llama3")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

    st.success("Ready! Ask your question below:")

    question = st.text_input("Ask a question about the PDF:")

    if question:
        with st.spinner("Thinking..."):
            answer = qa_chain.run(question)
        st.markdown("### 💬 Answer:")
        st.write(answer)

    os.unlink(tmp_path)
