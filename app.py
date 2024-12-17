import os
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import json
from PyPDF2 import PdfReader
import pickle

# Set Mistral API Key 
MISTRAL_API_KEY = "API_KEY"

# Initialize SBERT model
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Faiss index
def create_faiss_index(dim=384):
    index = faiss.IndexFlatL2(dim)  # Using L2 distance for vector search
    return index

# Function to generate SBERT embeddings
def generate_sbert_embeddings(texts):
    embeddings = sbert_model.encode(texts)
    return embeddings

# Function to query Mistral API
def query_mistral(prompt, api_key):
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "mistral-large-latest",
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raises HTTPError for bad responses
        response_data = response.json()
        return response_data['choices'][0]['message']['content']
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except requests.exceptions.RequestException as req_err:
        return f"Request error occurred: {req_err}"
    except KeyError:
        return "Unexpected response structure from Mistral API."

# Function to chunk text into tokens
def tokenize_and_chunk(text, chunk_size=1):
    tokens = text.split()  # Split text into individual tokens
    chunks = [" ".join(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size)]
    return chunks

# Streamlit App
st.title("PDF Query Application with Token Storage and Embeddings in FAISS")

# Sidebar for Project Management
st.sidebar.header("Manage Projects")

# Create a New Project
project_name = st.sidebar.text_input("Enter Project Name:")
if st.sidebar.button("Create Project"):
    project_dir = f"projects/{project_name}"
    os.makedirs(project_dir, exist_ok=True)
    st.session_state.projects[project_name] = {
        "tokens": [],
        "index_path": f"{project_dir}/index.faiss",
        "tokens_path": f"{project_dir}/tokens.pkl"
    }
    st.sidebar.success(f"Project '{project_name}' created.")

# List Existing Projects
if "projects" not in st.session_state:
    st.session_state.projects = {}

projects = list(st.session_state.projects.keys())
selected_project = st.sidebar.selectbox("Select a Project", projects)

# Upload Multiple PDFs for Selected Project
if selected_project:
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    project_data = st.session_state.projects[selected_project]
    
    if uploaded_files:
        project_dir = f"projects/{selected_project}"
        os.makedirs(project_dir, exist_ok=True)

        # Load existing embeddings and tokens
        if os.path.exists(project_data["index_path"]):
            index = faiss.read_index(project_data["index_path"])
        else:
            index = create_faiss_index()

        if os.path.exists(project_data["tokens_path"]):
            with open(project_data["tokens_path"], "rb") as f:
                tokens = pickle.load(f)
        else:
            tokens = []

        new_tokens = []
        for uploaded_file in uploaded_files:
            pdf_reader = PdfReader(uploaded_file)
            full_text = " ".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
            # Tokenize the extracted text
            chunks = tokenize_and_chunk(full_text, chunk_size=1)
            new_tokens.extend(chunks)

        embeddings = generate_sbert_embeddings(new_tokens)
        index.add(np.array(embeddings).astype(np.float32))  # Adding to Faiss index
        tokens.extend(new_tokens)

        # Save updated index and tokens
        faiss.write_index(index, project_data["index_path"])
        with open(project_data["tokens_path"], "wb") as f:
            pickle.dump(tokens, f)

        st.session_state.projects[selected_project]["tokens"] = tokens
        st.success(f"{len(uploaded_files)} PDFs processed and tokens stored.")

# Query PDFs
if selected_project:
    question = st.text_input("Enter your question:")
    if question:
        project_data = st.session_state.projects[selected_project]

        # Load Faiss index and tokens
        index = faiss.read_index(project_data["index_path"])
        with open(project_data["tokens_path"], "rb") as f:
            tokens = pickle.load(f)

        # Generate embedding for the question
        query_embedding = generate_sbert_embeddings([question])[0].reshape(1, -1)

        # Search embeddings in the Faiss index
        D, I = index.search(query_embedding, k=5)  # k=5 for top 5 results

        # Retrieve the corresponding tokens
        retrieved_tokens = "\n".join([tokens[i] for i in I[0] if i < len(tokens)])

        # Use Mistral API to refine answer based on retrieved tokens
        prompt = f"Only Based on the following information:\n{retrieved_tokens}\nAnswer the question: {question}"
        mistral_answer = query_mistral(prompt, MISTRAL_API_KEY)

        # Display Results
        st.write("Answer:")
        st.write(mistral_answer)
