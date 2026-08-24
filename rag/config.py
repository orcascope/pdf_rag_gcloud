PROJECT_ID = 'farmlink-506210'
DATASET_ID = 'sample_doc'
LOCATION = 'us'
VERTEX_LOCATION = 'us-central1'

CONNECTION_ID = f'{PROJECT_ID}.{LOCATION}.docai_connection'
PROCESSOR_ID = '6b66b64e18062e8d'
PROCESSOR = f'projects/{PROJECT_ID}/locations/{LOCATION}/processors/{PROCESSOR_ID}'

SOURCE_URIS = ['gs://rawdata_minio/pdf/*']

DOCUMENTS_TABLE = f'{PROJECT_ID}.{DATASET_ID}.documents_table'
DOCAI_MODEL = f'{PROJECT_ID}.{DATASET_ID}.docai_model'
PARSED_TABLE = f'{PROJECT_ID}.{DATASET_ID}.documents_parsed'
EMBDING_TABLE = f'{PROJECT_ID}.{DATASET_ID}.document_embeddings'
SKIP_ALREADY_PARSED = True

EMBDING_MODEL = "text-embedding-005"
GENERATIVE_MODEL = "gemini-2.5-flash"

from pathlib import Path
OUTPUT_PATH = Path(__file__).resolve().parents[2] / 'content' / 'processed_docs.json'

###Prepares the chunks of rags reading text from bq table.


PAGE_SIZE = 100
CHUNK_SIZE = 1000
OVERLAP = 200
