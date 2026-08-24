from rag.schemas import QueryRequest, QueryResponse
from rag.config import *
from rag.generation import generate_response
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPICallError
from datetime import datetime
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel,GenerationConfig
from contextlib import asynccontextmanager
import vertexai
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    vertexai.init(project=PROJECT_ID, location=VERTEX_LOCATION)
    app.state.bq = bigquery.Client(project=PROJECT_ID)
    app.state.embedder = TextEmbeddingModel.from_pretrained(EMBDING_MODEL)
    app.state.llm = GenerativeModel(GENERATIVE_MODEL)
    yield                      # app serves requests here
    app.state.bq.close()

app = FastAPI(lifespan=lifespan)

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) ->QueryResponse:
    answer = generate_response(req.query)
    return QueryResponse(answer=answer)

@app.get("/health")
def health():
    return {"status" : "ok"}

