from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)

class QueryResponse(BaseModel):
    answer: str | None    

class Source(BaseModel):
    doc_id: str
    content: str
    score: float

