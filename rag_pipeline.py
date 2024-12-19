import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss  # For vector database
from transformers import pipeline

# Load the PDF file
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# Chunk the text into smaller segments
def chunk_text(text, chunk_size=512):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Embed the chunks using a pre-trained model
def embed_chunks(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(chunks)

# Store embeddings in a vector database
def store_embeddings(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance
    index.add(embeddings)
    return index

# Convert user query into embeddings
def embed_query(query):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode([query])

# Perform similarity search
def search_similar_chunks(query_embedding, index, chunks, top_k=5):
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

# Function to handle comparison queries
def handle_comparison_query(query, index, chunks):
    relevant_chunks = search_similar_chunks(embed_query(query), index, chunks)
    return relevant_chunks

# Generate response using LLM
def generate_response(retrieved_chunks, user_query):
    llm = pipeline("text-generation", model="gpt-3.5-turbo")  # Example model
    context = "\n".join(retrieved_chunks)
    prompt = f"Answer the following question based on the context:\n{context}\nQuestion: {user_query}\nAnswer:"
    response = llm(prompt, max_length=150)
    return response[0]['generated_text']

# Main function to execute the pipeline
def main(pdf_path, user_query):
    # Data ingestion
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    index = store_embeddings(embeddings)

    # Query handling
    retrieved_chunks = search_similar_chunks(embed_query(user_query), index, chunks)

    # Response generation
    response = generate_response(retrieved_chunks, user_query)
    print(response)

# Example usage
if __name__ == "__main__":
    pdf_path = "C:/Users/Vishpath/Downloads/Tables, Charts, and Graphs with Examples from History, Economics, Education, Psychology, Urban Affairs and Everyday Life - 2017-2018.pdf"
    user_query = "From page 2 get the exact unemployment information based on type of degree input" 
    main(pdf_path, user_query)
