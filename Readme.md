# pdf_rag_gcloud

A document Q&A service over a bucket of PDFs. Ask a question in natural language and
get an answer grounded only in those documents, with the source chunks it used.

The pipeline is BigQuery-native: OCR, vector storage, and similarity search all happen
against BigQuery rather than a separate vector database. Document AI is invoked from
SQL through a remote model, so parsed text lands in a table without ever passing
through Python.

## Architecture

```
   GCS bucket
   (gs://.../pdf/*)
        │
        │  hop 1 - ingest
        ▼
   documents_table  ──▶  ML.PROCESS_DOCUMENT  ──▶  documents_parsed
   (object table)        (Document AI remote          (OCR JSON per file)
                          model)
        │
        │  hop 2 - chunk
        ▼
   overlapping text chunks + metadata
        │
        │  hop 3 - embed
        ▼
   document_embeddings
   (doc_id, source, content, embedding ARRAY<FLOAT64>)
        │
        │  hop 4 - retrieve + generate
        ▼
   cosine top-k  ──▶  grounded prompt  ──▶  Gemini  ──▶  FastAPI /query
```

## The four hops

### Hop 1 — Ingest and OCR (`rag/ingest.py`)

Three BigQuery objects are created, then one statement does the work.

An **object table** (`documents_table`) points at `gs://rawdata_minio/pdf/*`. It holds one
row per file — uri, size, content type, and a `ref` to the bytes — turning a bucket into
something queryable. A **remote model** (`docai_model`) points at a Document AI OCR
processor. `ML.PROCESS_DOCUMENT` then joins the two, sending each file to Document AI and
writing the OCR result into `documents_parsed` as JSON.

Access is delegated through a BigQuery connection whose service account holds
`documentai.viewer` and `storage.objectViewer` — so the pipeline reads the bucket without
any caller needing direct GCS access.

`SKIP_ALREADY_PARSED` swaps the input from the whole object table to a subquery excluding
files already parsed successfully. Document AI bills per page, so re-running the pipeline
must not re-OCR work that is already done.

### Hop 2 — Chunk (`rag/chunking.py`)

OCR text is split into overlapping chunks of `CHUNK_SIZE` characters with `OVERLAP`
characters carried between them, so a sentence spanning a boundary still appears intact in
one of the two chunks.

`chunk_text` prefers to break at a sentence or paragraph boundary, searching backwards from
the target end within the last quarter of the chunk. `prepare_for_rag` then wraps each chunk
with the metadata retrieval needs: a stable `id`, the source uri, chunk index, and character
count.

### Hop 3 — Embed and store (`rag/get_embeddings.py`)

Each chunk is embedded with Vertex AI `text-embedding-005` and written to
`document_embeddings` alongside its text and source.

The embedding column is `ARRAY<FLOAT64>` — the type BigQuery's vector functions and vector
indexes require, so the storage format does not need to change when retrieval moves
server-side.

### Hop 4 — Retrieve and generate (`rag/retrieval.py`, `rag/generation.py`)

The question is embedded with the same model, then scored by cosine similarity against every
stored chunk. Rows stream a page at a time and a min-heap keeps only the best `top_k`, so
memory stays flat regardless of corpus size.

The winning chunks are assembled into a prompt instructing the model to answer only from the
provided context, and Gemini generates the answer. Grounding is the point: the sources come
back with the response so an answer can be traced to a document.

## Project layout

```
rag/            importable library - one module per hop
  config.py       project, dataset, table, and model ids
  clients.py      BigQuery and Vertex client construction
  ingest.py       hop 1
  chunking.py     hop 2
  get_embeddings.py  hop 3
  retrieval.py    hop 4 - vector search
  generation.py   hop 4 - prompt assembly and Gemini
  api.py          FastAPI app, lifespan, routes
  schemas.py      pydantic request/response models
pipelines/      thin entry points, numbered by run order
doc/            scratch notes and earlier flat versions
```

The library carries stable names; the numeric ordering lives on the entry points, which are
the things actually run in sequence.

## Running it

Install once so `rag` is importable from anywhere:

```bat
pip install -e .
```

Batch pipeline — parse PDFs, chunk, embed:

```bat
python -m pipelines.01_ingest_embed
```

One-off query from the command line:

```bat
python -m pipelines.02_use_rag
```

Serve the API:

```bat
uvicorn rag.api:app --reload --port 8080
```

`http://localhost:8080/docs` gives an interactive UI generated from the pydantic schemas.

## Configuration

All ids live in `rag/config.py`. Note that "location" means three different things with
incompatible value sets:

| Setting | Valid values | Used by |
|---|---|---|
| `LOCATION` | `US`, `EU`, `us-central1` | BigQuery dataset, connection |
| Document AI processor region | `us`, `eu` only | `PROCESSOR` |
| `VERTEX_LOCATION` | `us-central1`, `europe-west4`, … | `vertexai.init` |

Vertex AI does not accept `us`; passing it silently routes to the global endpoint, where the
publisher models are not served.

## Known gaps

- **Retrieval is O(corpus) per query.** Every search pulls all embeddings to the client.
  The next change is a BigQuery vector index with `VECTOR_SEARCH`, moving the scan
  server-side and making it approximate.
- **Re-running embedding duplicates rows.** `document_embeddings` has no uniqueness
  constraint; a per-source delete before insert, or a load job with `WRITE_TRUNCATE`,
  would make it idempotent.
- **Retrieved text is interpolated into the prompt undelimited**, so a document containing
  instructions can influence the model. Retrieved content needs to be fenced and framed as
  untrusted data.
- **No relevance threshold.** The top-k chunks are used even when every score is low, so an
  off-topic question still gets confident-sounding context.
- **No tests or evals**, no structured logging, and infrastructure (connection, processor,
  dataset, IAM) was created by hand rather than declared in Terraform.
