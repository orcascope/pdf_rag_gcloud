from rag.config import *
from rag.clients import bq_client
from datetime import datetime

def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Split text into overlapping chunks.
    
    Args:
        text: The text to chunk
        chunk_size: Target size of each chunk in characters
        overlap: Number of overlapping characters between chunks
    
    Returns:
        List of text chunks
    """
    if overlap >= chunk_size:
        raise ValueError('overlap must be smaller than chunk_size')

    chunks = []
    start = 0
    length = len(text)

    while start < length:
        # Find end of chunk
        end = min(start + chunk_size, length)

        # Try to break at a sentence boundary
        if end < length:
            # Look in the last quarter of the chunk only, so `end` always
            # stays ahead of `start` no matter how small chunk_size is
            window = max(start + 1, end - chunk_size // 4)
            for sep in ['. ', '\n\n', '\n', ' ']:
                boundary = text.rfind(sep, window, end)
                if boundary != -1:
                    end = boundary + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= length:
            break

        # Move start position with overlap, always making forward progress
        start = max(end - overlap, start + 1)

    return chunks

def docid_exist_in_embeddings_table(doc_id):
    table_ref = EMBDING_TABLE
    sql = f"""SELECT 1 FROM {table_ref} WHERE DOC_ID = '{doc_id}' """
    res = bq_client.query_and_wait(sql)
    if res.total_rows > 0:
        return True
    return False


def prepare_for_rag(chunks, document_uri, metadata=None):
    """
    Prepare document chunks for a RAG system.
    
    Args:
        chunks: List of text chunks
        document_uri: Source document URI
        metadata: Optional document metadata
    
    Returns:
        List of RAG-ready document objects
    """
    rag_documents = []
    
    for i, chunk in enumerate(chunks):
        doc = {
            "id": f"{document_uri.split('/')[-1]}_{i}",
            "content": chunk,
            "source": document_uri,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "char_count": len(chunk),
            "processed_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        if not docid_exist_in_embeddings_table(doc["id"]):
            rag_documents.append(doc)
    
    return rag_documents

