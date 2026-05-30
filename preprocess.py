import re
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

def clean_text(text: str) -> str:
    """
    Cleans raw text by:
    1. Removing HTML tags.
    2. Fixing character encoding or special character quirks.
    3. Normalizing whitespace (replacing tabs, multiple spaces, multiple newlines).
    """
    if not text:
        return ""
    
    # Remove HTML tags using regex
    cleaned = re.sub(r'<[^>]*>', ' ', text)
    
    # Normalize smart quotes and special dash characters
    cleaned = cleaned.replace('\u201c', '"').replace('\u201d', '"')
    cleaned = cleaned.replace('\u2018', "'").replace('\u2019', "'")
    cleaned = cleaned.replace('\u2013', '-').replace('\u2014', '-')
    
    # Replace multiple whitespaces/tabs with a single space
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    
    # Replace multiple newlines with a single newline
    cleaned = re.sub(r'\n+', '\n', cleaned)
    
    # Strip leading/trailing whitespaces
    return cleaned.strip()

def chunk_document(doc: Dict[str, Any], chunk_size: int = 400, chunk_overlap: int = 80) -> List[Dict[str, Any]]:
    """
    Cleans a document and splits its content into overlapping chunks.
    Preserves document metadata (id, hotel, category, title) on each chunk.
    """
    raw_content = doc.get("content", "")
    cleaned_content = clean_text(raw_content)
    
    # Format a header that includes metadata to inject context directly into the text chunk
    # This helps the embedding model associate the text with the specific hotel and category.
    metadata_prefix = f"Hotel: {doc.get('hotel')} | Category: {doc.get('category')} | Title: {doc.get('title')} | Content: "
    
    # Initialize the splitter
    # Note: We split the cleaned content, and then append metadata to the text of the chunks.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""]
    )
    
    splits = splitter.split_text(cleaned_content)
    
    chunks = []
    for idx, split in enumerate(splits):
        # We store both the raw split content and a fully-contextualized content string for embedding.
        # Placing metadata inside the chunk content boosts similarity matching.
        contextualized_content = f"Hotel: {doc.get('hotel')} | Category: {doc.get('category')} | {split}"
        
        chunks.append({
            "chunk_id": f"{doc.get('id')}-C{idx+1}",
            "doc_id": doc.get("id"),
            "hotel": doc.get("hotel"),
            "category": doc.get("category"),
            "title": doc.get("title"),
            "content": split,
            "contextualized_content": contextualized_content
        })
        
    return chunks

def preprocess_dataset(dataset_path: str, chunk_size: int = 400, chunk_overlap: int = 80) -> List[Dict[str, Any]]:
    """
    Loads the JSON dataset, cleans each document, chunks them, and returns a list of chunk dictionaries.
    """
    import json
    with open(dataset_path, "r", encoding="utf-8") as f:
        docs = json.load(f)
        
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, chunk_size, chunk_overlap))
        
    return all_chunks

if __name__ == "__main__":
    import json
    # Simple self-test
    test_doc = {
        "id": "DOC-TEST",
        "hotel": "Hotel Test",
        "category": "Policies",
        "title": "Cancellation Policy",
        "content": "<p>This is a <b>test</b> of the cleaning and chunking system.</p> It should clean HTML tags and segment this into chunks properly."
    }
    chunks = chunk_document(test_doc, chunk_size=100, chunk_overlap=20)
    print("Test Cleaning & Chunking output:")
    print(json.dumps(chunks, indent=2))
