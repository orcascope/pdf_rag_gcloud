from rag.config import *
from rag.clients import *
from rag.chunking import chunk_text,prepare_for_rag

def read_rows(project_id=PROJECT_ID,
              dataset_id=DATASET_ID,
              table_id=PARSED_TABLE,
              where=None,
              page_size=PAGE_SIZE,
              max_rows=None):
    """Yield one row at a time from a BigQuery table.

    Rows are streamed a page at a time, so the whole table is never held
    in memory. With no `where` this is a free table read; passing `where`
    runs a query instead, which is billed on the bytes scanned.
    """
    if where:
        sql = f"SELECT * FROM `{table_id}` WHERE {where}"
        rows = bq_client.query(sql).result(page_size=page_size, max_results=max_rows)
    else:
        rows = bq_client.list_rows(table_id, page_size=page_size, max_results=max_rows)

    for row in rows:
        yield row


def process_row(row):
    """Downstream processing hook - called once per row."""
    result = row['ml_process_document_result']
    pdf_uri = row['uri']

    # JSON columns come back parsed on newer clients, as a string on older ones
    if isinstance(result, str):
        result = json.loads(result)

    text = (result or {}).get('text', '')
    rag_docs = []
    if len(text) >0:
        chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)

        # @title Preview chunks
        print("Sample Chunks:\n")
        for i, chunk in enumerate(chunks[:3]):
            print(f"--- Chunk {i+1} ({len(chunk)} chars) ---")
            print(chunk[:300] + "..." if len(chunk) > 300 else chunk)
            print()

        # Prepare our chunks for RAG
        rag_docs = prepare_for_rag(
            chunks,
            pdf_uri,
            metadata=None
        )
        print(len(rag_docs), "rag doc length")
        generate_embedding(rag_docs)

    print(f"{row['uri']} -> {len(text)} chars, {len(rag_docs)} chunks")
    return rag_docs


def run(handler=process_row, **kwargs):
    """Read the table row by row and hand each row to `handler`."""
    count = 0
    rag_docs = []
    for row in read_rows(**kwargs):
        rag_docs.extend(handler(row) or [])
        count += 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(rag_docs, f, indent=2, default=str)

    print(f'processed {count} rows -> {len(rag_docs)} chunks in {OUTPUT_PATH}')
    return rag_docs


def generate_embedding(rag_docs:list)->list:
    """Generates embeddings and inserts into document_embeddings bq table"""  
    rows_to_insert = [
                        {
                            "doc_id" : doc.get("id"),
                            "source" : doc.get("source"),
                            "content" : doc.get("content"),
                            "embedding" : embedding_model.get_embeddings([doc.get("content")])[0].values                                            
                        }
                        for doc in rag_docs
    ]
    if rows_to_insert:
        bq_client.insert_rows_json(EMBDING_TABLE, rows_to_insert)
    print(f"{len(rows_to_insert)} embeddings inserted into {EMBDING_TABLE}")


if __name__ == '__main__':
    run(where="ml_process_document_status = ''", max_rows=5)