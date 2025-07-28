import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader
import cohere
from dotenv import load_dotenv
import time

# Load environment variables including GROQ_API_KEY and COHERE_API_KEY
load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.title("Llama-4-Scout RAG Chatbot")

# --- Check API keys loaded ---
if not GROQ_API_KEY:
    st.error("Please set the GROQ_API_KEY environment variable.")
    st.stop()
if not COHERE_API_KEY:
    st.error("Please set the COHERE_API_KEY environment variable.")
    st.stop()

# Initialize the Groq LLM with the correct model name (match Groq naming exactly)
# IMPORTANT: Confirm the model name from your Groq dashboard or docs. 
# Commonly it is 'llama-4-scout' or some variant — double-check exact spelling.
try:
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model="meta-llama/llama-4-scout-17b-16e-instruct")
except Exception as e:
    st.error(f"Error initializing the LLM: {e}")
    st.stop()

# Define prompt template for RAG
prompt = ChatPromptTemplate.from_template(
    """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Question: {input}
"""
)

# Initialize Cohere client and embedder wrapper
co = cohere.Client(COHERE_API_KEY)


class MyCohereEmbedder:
    def embed_documents(self, texts):
        response = co.embed(
            texts=texts,
            model="embed-english-v3.0",
            input_type="search_document"
        )
        return response.embeddings

    def embed_query(self, text):
        response = co.embed(
            texts=[text],
            model="embed-english-v3.0",
            input_type="search_query"
        )
        return response.embeddings[0]

    def __call__(self, text):
        return self.embed_query(text)


def vector_embeddings(user_url):
    """Load docs, split into chunks, embed, and build vector store."""
    if ("vectors" not in st.session_state) or (st.session_state.get("last_url") != user_url):
        st.session_state.last_url = user_url
        st.session_state.embeddings = MyCohereEmbedder()
        # Load documents from URL
        st.session_state.loader = WebBaseLoader(user_url)
        try:
            st.session_state.documents = st.session_state.loader.load()
        except Exception as e:
            st.error(f"Failed to load documents from URL: {e}")
            return
        # Split documents into chunks for embedding
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.documents)
        # Create FAISS vector store
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.success("Vector Store DB is ready.")


# Streamlit UI
user_url = st.text_input("Enter the website URL to load content from")
prompt1 = st.text_input("Enter your question from documents")

if st.button("Create Document Embeddings") and user_url:
    vector_embeddings(user_url)

if prompt1 and ("vectors" in st.session_state):
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    try:
        response = retrieval_chain.invoke({"input": prompt1})
        elapsed = time.process_time() - start
        st.write(f"Response Time: {elapsed:.2f} seconds")
        st.write("### Answer:")
        st.write(response['answer'])

        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response.get("context", [])):
                st.write(doc.page_content)
                st.write("---")
    except Exception as e:
        st.error(f"Failed to get LLM response: {e}")
