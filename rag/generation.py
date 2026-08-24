from rag.retrieval import vector_search
from rag.config import *
from rag.clients import generative_model, generation_config


def assemble_rag_context(query):
    "Perform vector search and get the related docs from doc_embeddings"
    retrieved_docs = vector_search(query)

    if retrieved_docs:
        retrieved_docs = sorted(retrieved_docs, key=lambda x:x[0], reverse=True) 
        prompt= f""" You are a helpful assistant. Look into the given context and answer the question 
                    grounded to the available context information. Use the context to provide a meaningful formatted response based on the question.
                    context: { ("\n")
                              .join([f"{doc[3]}  |Doc source: {doc[1]}|" for doc in retrieved_docs ])
                    }
                    query: {query}
                """
        return prompt
    return None

def generate_response(query):
    prompt = assemble_rag_context(query)
    response = generative_model.generate_content(prompt, generation_config =generation_config)
    print(response.text) 
    return response.text

if __name__ == "__main__":
    generate_response("Is Spark used along with PowerBI")    

