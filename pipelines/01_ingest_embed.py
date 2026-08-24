from rag.ingest import main
from rag.get_embeddings import run

main()
run(where="ml_process_document_status = ''", max_rows=5)
