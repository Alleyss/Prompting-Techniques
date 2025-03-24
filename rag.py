
import faiss
import json
import os
import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv
load_dotenv()
# Set your Gemini API Key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# FAISS Index Path
index_path = "faiss_index"

import numpy as np
import google.generativeai as genai

def get_google_embeddings(text):
    """Get embeddings from Google's embedding model (models/embedding-001)."""
    model = genai.GenerativeModel("models/embedding-001")  # Correct model
    response = genai.embed_content("models/text-embedding-004", text, task_type="retrieval_document")
    return response["embedding"]  # Extract embedding vector


def create_faiss_index(text_chunks):
    """Create FAISS index from text chunks using Google Embeddings."""
    embeddings = [get_google_embeddings(text) for text in text_chunks]

    # Create FAISS index
    d = len(embeddings[0])  # Embedding dimension
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings, dtype="float32"))

    # Save index
    faiss.write_index(index, index_path)

    # Save text data
    with open("text_data.json", "w") as f:
        json.dump(text_chunks, f)

    print("FAISS index created and saved.")

def load_faiss_index():
    """Load FAISS index and text chunks."""
    if not os.path.exists(index_path):
        raise FileNotFoundError("FAISS index not found. Create it first.")

    index = faiss.read_index(index_path)
    with open("text_data.json", "r") as f:
        text_chunks = json.load(f)

    return index, text_chunks

def retrieve_documents(query, top_k=3):
    """Retrieve top-k relevant documents using FAISS."""
    index, text_chunks = load_faiss_index()

    query_embedding = np.array([get_google_embeddings(query)], dtype="float32")
    distances, indices = index.search(query_embedding, top_k)

    retrieved_texts = [text_chunks[i] for i in indices[0] if i < len(text_chunks)]
    return retrieved_texts

def generate_response(query):
    """Generate response using retrieved documents and Gemini API."""
    retrieved_docs = retrieve_documents(query)

    # Construct prompt for Gemini
    context = "\n".join(retrieved_docs)
    prompt = f"""
    You are an AI assistant. Use the following retrieved documents to answer the query.

    Context:
    {context}

    Query: {query}

    Answer:
    """

    # Call Gemini API
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    return response.text

# Example Usage
if __name__ == "__main__":
    # Example dataset
    # documents = [
    #     "RAG stands for Retrieval-Augmented Generation.",
    #     "FAISS is an efficient similarity search library for embeddings.",
    #     "LangChain is a framework for building LLM-powered applications."
    # ]
    documents= [
    "The Grand Oak Tree, standing for over 300 years, provides shelter to countless birds and animals.",
    "Alexander Graham Bell invented the telephone in 1876, revolutionizing communication.",
    "The Great Wall of China, built primarily during the Ming Dynasty, stretches over 13,000 miles.",
    "The Amazon Rainforest, known as the lungs of the Earth, produces 20% of the world’s oxygen.",
    "The Eiffel Tower, constructed in 1889, stands at 330 meters tall and attracts millions of visitors annually.",
    "Albert Einstein developed the theory of relativity in 1905, which changed modern physics.",
    "The Pacific Ocean, covering over 63 million square miles, is the largest and deepest ocean on Earth.",
    "The Sahara Desert, spanning 9.2 million square kilometers, is the largest hot desert in the world.",
    "William Shakespeare, born in 1564, wrote famous plays like Hamlet, Macbeth, and Romeo & Juliet.",
    "The Moon Landing on July 20, 1969, saw Neil Armstrong become the first human to walk on the moon.",
    "Mount Everest, standing at 8,848.86 meters, is the highest mountain in the world, located in Nepal.",
    "The Pyramids of Giza, built around 2,500 BC, are one of the Seven Wonders of the Ancient World.",
    "The Taj Mahal, a symbol of love, was built in 1632 by Shah Jahan in memory of his wife.",
    "The Blue Whale, the largest animal on Earth, can grow up to 30 meters long and weigh 200 tons.",
    "Japan, known as the Land of the Rising Sun, consists of 6,852 islands, with Tokyo as its capital.",
    "The Statue of Liberty, a gift from France, was unveiled in 1886 and symbolizes freedom and democracy.",
    "The Hubble Space Telescope, launched in 1990, has provided breathtaking images of the universe.",
    "The Wright Brothers, in 1903, achieved the first powered flight with their aircraft, Flyer I.",
    "Leonardo da Vinci, a Renaissance genius, painted the Mona Lisa and designed early flying machines.",
    "The Golden Gate Bridge, completed in 1937, spans 1.7 miles across the San Francisco Bay."
]

    # Create FAISS index
    create_faiss_index(documents)

    # Query the system
    query = "What happened in 1937?"
    answer = generate_response(query)
    print("AI Response:", answer)
