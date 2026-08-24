from rag.config import *
from rag.clients import *
import numpy as np
import heapq

def cosine_similiarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def read_embeddings(table_id = EMBDING_TABLE, page_size=PAGE_SIZE):
    """Yield one stored embedding at a time.

    The client fetches a page at a time, so only `page_size` rows are held
    in memory no matter how large the table grows.
    """
    sql = f"SELECT doc_id, source, content, embedding FROM `{table_id}`"

    for row in bq_client.query_and_wait(sql, page_size=page_size):
        yield row


def vector_search(query, top_k=5)->list:
    """Score every stored chunk against `query` and return the best `top_k`."""

    query_embedding = embedding_model.get_embeddings([query])[0].values
    query_vec = np.array(query_embedding)

    # A min-heap of the best matches so far - only top_k rows are ever kept
    best = []

    for row in read_embeddings():
        score = cosine_similiarity(query_vec, np.array(row['embedding']))
        match = (float(score), row['doc_id'], row['source'], row['content'])

        if len(best) < top_k:
            heapq.heappush(best, match)
        elif score > best[0][0]:
            heapq.heapreplace(best, match)

    return sorted(best, reverse=True)


if __name__ == '__main__':
    results = vector_search('data engineering experience with spark and kafka')

    for score, doc_id, source, content in results:
        print(f"{score:.4f}  {doc_id}")
        print(content[:200].replace('\n', ' '))
        print()
