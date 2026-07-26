from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.graph_db import KnowledgeGraphEngine
from src.vector_db import VectorSearchEngine 
from src.entity_extraction import extract_triplets

app = FastAPI(title="BioCausal-RAG Engine", version="1.0.0")

# Initialize database engines
graph_engine = KnowledgeGraphEngine()
vector_engine = VectorSearchEngine()

class SeedDataRequest(BaseModel):
    text_chunks: list[str]

class QueryRequest(BaseModel):
    query: str
    target_entity: str

@app.get("/")
def root():
    return {"status": "BioCausal-RAG Engine is live and operational"}

@app.post("/ingest")
async def ingest_text_data(request: SeedDataRequest):
    """Ingests raw scientific text into both Qdrant (Vector) and Neo4j (Graph)."""
    try:
        # 1. Store in Qdrant Vector DB
        vector_engine.add_chunks(request.text_chunks)
        
        # 2. Extract triplets & insert into Neo4j Graph DB
        extracted_count = 0
        for chunk in request.text_chunks:
            triplets = extract_triplets(chunk)
            for subj, rel, obj in triplets:
                graph_engine.insert_triplet(subj, rel, obj)
                extracted_count += 1
                
        return {
            "status": "success",
            "chunks_indexed": len(request.text_chunks),
            "graph_triplets_created": extracted_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def execute_hybrid_query(request: QueryRequest):
    """Executes hybrid retrieval: Vector similarity search + Graph path traversal."""
    try:
        # 1. Vector Search
        text_chunks = vector_engine.search(request.query, limit=2)
        
        # 2. Graph Traversal (Sub-graph reasoning)
        graph_paths = graph_engine.k_hop_subgraph(request.target_entity, k=2)
        
        return {
            "query": request.query,
            "target_entity": request.target_entity,
            "graph_causal_paths": graph_paths,
            "retrieved_vector_chunks": text_chunks,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))