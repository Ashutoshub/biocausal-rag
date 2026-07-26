from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
class VectorSearchEngine:
    def __init__(self, collection_name="bio_papers"):
        # Connect to the local Qdrant container running on port 6333
        self.client = QdrantClient("localhost", port=6333)
        # Load lightweight MiniLM model for generating dense embeddings
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.collection_name = collection_name
        
        # Initialize collection if it doesn't exist yet
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )

    def add_chunks(self, chunks: list):
        """Encodes text chunks and inserts vector points into Qdrant."""
        points = []
        for idx, chunk in enumerate(chunks):
            vector = self.encoder.encode(chunk).tolist()
            points.append(PointStruct(id=idx, vector=vector, payload={"text": chunk}))
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query: str, limit: int = 3) -> list:
        """Searches vector space for top-k semantically similar text chunks."""
        query_vector = self.encoder.encode(query).tolist()
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit
        ).points
        return [res.payload["text"] for res in results]