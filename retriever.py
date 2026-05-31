import os
from typing import List, Dict, Any, Tuple
from settings import FAISS_INDEX_PATH, BASE_DIR
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

# Embedding backends configuration
def get_embeddings(backend: str = "huggingface", openai_api_key: str = None):
    """
    Returns the appropriate embedding model based on the selected backend.
    """
    backend = backend.lower()
    if backend == "ollama":
        # Import dynamically to avoid loading times if not used
        from langchain_ollama import OllamaEmbeddings
        print("Using local Ollama embeddings (nomic-embed-text)...")
        return OllamaEmbeddings(model="nomic-embed-text")
    
    elif backend == "openai":
        from langchain_openai import OpenAIEmbeddings
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "dummy_key")
        print("Using OpenAI embeddings (text-embedding-3-small)...")
        return OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    
    else:  # default to huggingface sentence-transformers
        from langchain_community.embeddings import HuggingFaceEmbeddings
        print("Using local Hugging Face embeddings (sentence-transformers/all-MiniLM-L6-v2)...")
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_vector_store(chunks: List[Dict[str, Any]], index_path: str = None, backend: str = "huggingface", openai_api_key: str = None):
    """
    Takes preprocessed chunks, generates embeddings, builds a FAISS index, and saves it.
    Uses 'contextualized_content' for the vector representations.
    """
    # resolve default index path from settings if not provided
    index_path = index_path or FAISS_INDEX_PATH
    embeddings_model = get_embeddings(backend, openai_api_key)
    
    documents = []
    for chunk in chunks:
        # We index using the contextualized content, but retain raw fields in metadata
        doc = Document(
            page_content=chunk["contextualized_content"],
            metadata={
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "hotel": chunk["hotel"],
                "category": chunk["category"],
                "title": chunk["title"],
                "content": chunk["content"]  # verbatim chunk content (uncontextualized)
            }
        )
        documents.append(doc)
        
    print(f"Embedding and indexing {len(documents)} chunks in FAISS...")
    vector_store = FAISS.from_documents(documents, embeddings_model)
    
    # Save the index to disk
    os.makedirs(index_path, exist_ok=True)
    vector_store.save_local(index_path)
    print(f"FAISS index successfully saved to: {index_path}")
    return vector_store

def load_vector_store(index_path: str = None, backend: str = "huggingface", openai_api_key: str = None):
    """
    Loads an existing FAISS index from disk.
    """
    index_path = index_path or FAISS_INDEX_PATH
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index folder not found at: {index_path}")
        
    embeddings_model = get_embeddings(backend, openai_api_key)
    # allow_dangerous_deserialization=True is required to load FAISS pickles locally
    vector_store = FAISS.load_local(index_path, embeddings_model, allow_dangerous_deserialization=True)
    return vector_store

def retrieve_chunks(query: str, index_path: str = None, backend: str = "huggingface", k: int = 3, openai_api_key: str = None) -> List[Dict[str, Any]]:
    """
    Retrieves the top-k chunks for a given query along with their distance scores.
    Returns a list of dictionaries containing chunk details and scores.
    """
    vector_store = load_vector_store(index_path, backend, openai_api_key)
    
    # similarity_search_with_score returns Tuple[Document, float] (L2 distance in FAISS; lower is closer)
    results: List[Tuple[Document, float]] = vector_store.similarity_search_with_score(query, k=k)
    
    retrieved_chunks = []
    for doc, distance in results:
        retrieved_chunks.append({
            "chunk_id": doc.metadata.get("chunk_id"),
            "doc_id": doc.metadata.get("doc_id"),
            "hotel": doc.metadata.get("hotel"),
            "category": doc.metadata.get("category"),
            "title": doc.metadata.get("title"),
            "verbatim_content": doc.metadata.get("content"),
            "contextualized_content": doc.page_content,
            "distance": float(distance)
        })
        
    return retrieved_chunks

if __name__ == "__main__":
    # Test building and retrieving from the vector store
    from preprocess import preprocess_dataset
    import json
    
    dataset_json = os.path.join(BASE_DIR, "hotel_dataset.json")
    if not os.path.exists(dataset_json):
        print("Please run generate_dataset.py first!")
    else:
        chunks = preprocess_dataset(dataset_json)
        # Build using local Hugging Face model
        build_vector_store(chunks, backend="huggingface")
        
        # Test retrieval
        test_query = "What is the cancellation policy of Hotel X?"
        results = retrieve_chunks(test_query, backend="huggingface", k=2)
        print("\nTest Retrieval Results:")
        print(json.dumps(results, indent=2))
