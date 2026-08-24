from google.api_core.exceptions import GoogleAPICallError
from google.cloud import bigquery
from rag.config import *
import json
from datetime import datetime
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel,GenerationConfig

bq_client = bigquery.Client(project=PROJECT_ID)
embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
generation_config = GenerationConfig(temperature=0, max_output_tokens=1024)
generative_model = GenerativeModel("gemini-2.5-flash")