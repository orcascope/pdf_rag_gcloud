from rag.config import *
from rag.clients import bq_client
from google.api_core.exceptions import GoogleAPICallError


URIS_SQL = ', '.join(f"'{uri}'" for uri in SOURCE_URIS)

CREATE_OBJECT_TABLE = f"""
CREATE EXTERNAL TABLE  IF NOT EXISTS  `{DOCUMENTS_TABLE}`
WITH CONNECTION `{CONNECTION_ID}`
OPTIONS (
    object_metadata = 'SIMPLE',
    uris = [{URIS_SQL}]
)
"""

CREATE_DOCAI_MODEL = f"""
CREATE MODEL  IF NOT EXISTS `{DOCAI_MODEL}`
REMOTE WITH CONNECTION `{CONNECTION_ID}`
OPTIONS (
    remote_service_type = 'CLOUD_AI_DOCUMENT_V1',
    document_processor = '{PROCESSOR}'
)
"""

CREATE_PARSED_TABLE = f"""
CREATE TABLE IF NOT EXISTS `{PARSED_TABLE}` (
  uri                        STRING    NOT NULL,
  ml_process_document_result JSON,
  ml_process_document_status STRING,
  processed_at               TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
OPTIONS (description = 'Document AI OCR output, one row per file per run')
"""

# Either every file, or only the ones with no successful row yet. The
# subquery must keep SELECT * so the `ref` column reaches Document AI.
if SKIP_ALREADY_PARSED:
    DOCUMENT_SOURCE = f"""(
        SELECT * FROM `{DOCUMENTS_TABLE}`
        WHERE uri NOT IN (
            SELECT uri FROM `{PARSED_TABLE}` WHERE ml_process_document_status = ''
        )
    )"""
else:
    DOCUMENT_SOURCE = f'TABLE `{DOCUMENTS_TABLE}`'

INSERT_PARSED = f"""
INSERT INTO `{PARSED_TABLE}`
  (uri, ml_process_document_result, ml_process_document_status)
SELECT
    uri,
    ml_process_document_result,
    ml_process_document_status
FROM ML.PROCESS_DOCUMENT(
    MODEL `{DOCAI_MODEL}`,
    {DOCUMENT_SOURCE}
)
"""

ddl_embedding_table = f"""create table if not exists  `{PROJECT_ID}.{DATASET_ID}.document_embeddings` 
                    (
                        doc_id STRING,
                        source STRING,
                        content STRING,
                        embedding ARRAY<FLOAT64>
                    )"""   
try:
    bq_client.query_and_wait(ddl_embedding_table)
except GoogleAPICallError as e:
    print(e , "DDL failed")
    raise


SUMMARY = f"""
SELECT
    uri,
    ml_process_document_status AS status,
    LENGTH(JSON_VALUE(ml_process_document_result.text)) AS chars,
    processed_at
FROM `{PARSED_TABLE}`
ORDER BY processed_at DESC
"""


def run_sql(name, sql):
    """Run one statement and block until BigQuery finishes it."""
    print(f'-> {name}')

    job = bq_client.query(sql)
    try:
        job.result()
    except GoogleAPICallError as e:
        print(f'   failed: {e}')
        raise

    return job


def scalar(sql):
    """Run a query that returns a single value."""
    for row in bq_client.query_and_wait(sql):
        return row[0]


def main():
    run_sql('create object table', CREATE_OBJECT_TABLE)
    print(f'   {scalar(f"SELECT COUNT(*) FROM `{DOCUMENTS_TABLE}`")} files matched {SOURCE_URIS}')

    run_sql('create remote model', CREATE_DOCAI_MODEL)
    run_sql('create parsed table', CREATE_PARSED_TABLE)

    # Billed per page and processed in batch - this is the slow step
    job = run_sql('parse documents', INSERT_PARSED)
    print(f'   {job.num_dml_affected_rows} rows inserted')

    print('\ndocuments_parsed:')
    for row in bq_client.query_and_wait(SUMMARY):
        status = row['status'] or 'ok'
        print(f"  {row['processed_at']:%Y-%m-%d %H:%M}  {row['chars'] or 0:>7} chars  {status:<6} {row['uri']}")


if __name__ == '__main__':
    main()