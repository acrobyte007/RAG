**PDF Question Answering with SBERT, FAISS, and Mistral API**
This project allows users to upload PDF documents, extract meaningful chunks of text, generate embeddings using SBERT (all-MiniLM-L6-v2), store the embeddings in a FAISS index, and query the content using the Mistral API to generate answers to user questions.

***Features***<br>
**PDF Upload:**<br>
Upload multiple PDF files and split them into meaningful text chunks.
**Text Embedding:**<br>
Generate sentence embeddings using the all-MiniLM-L6-v2 model from Sentence Transformers.
**FAISS Index:** Store and search embeddings efficiently using FAISS for nearest neighbor retrieval.
**Mistral API Integration:** Send retrieved context to Mistral's API to generate accurate answers to user queries.
**Project Management:** Dynamically create, delete, and manage projects.
**Caching:** Cache previously asked questions for faster responses.

***Requirements***
Python 3.8 or higher
Docker (for containerized deployment)
An active Mistral API Key
***Setup***
**1. Clone the Repository**
**2. Install Dependencies**
Run the following command to install all required libraries:
pip install -r requirements.txt
***Environment Setup***
Create a .env file in the project root with your Mistral API key:

MISTRAL_API_KEY=mistral_api_key
***Run the Application***
Run the Streamlit app locally:
streamlit run app.py
