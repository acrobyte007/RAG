**RAG-based PDF QA System**
<br>
Features
<br>
PDF Upload:<br>
Upload multiple PDF files and split them into meaningful text chunks.

Text Embedding:<br>
Generate sentence embeddings using the all-MiniLM-L6-v2 model from Sentence Transformers.

FAISS Index:<br>
Store and search embeddings efficiently using FAISS for nearest neighbor retrieval.

Mistral API Integration:<br>
Send retrieved context to Mistral's API to generate accurate answers to user queries.

Project Management:<br>
Dynamically create, delete, and manage projects.
Caching:<br>
Cache previously asked questions for faster responses.
<br>
Requirements
<br>
Python 3.8 or higher
Docker (for containerized deployment)
An active Mistral API Key
<br>
Setup
<br>
1. Clone the Repository
git clone <your-repository-url>
<br>
2. Install Dependencies<br>
Run the following command to install all required libraries:<br>
pip install -r requirements.txt
<br>
Environment Setup
<br>
Create a .env file in the project root with your Mistral API key:<br>
MISTRAL_API_KEY=mistral_api_key
<br>
Run the Application
<br>
Run the Streamlit app locally:<br>
streamlit run app.py
<br>
