import random
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


class VectorSearchEngine:

  def __init__(self, collection_name="bio_papers"):
    self.collection_name = collection_name
    self.client = QdrantClient(":memory:")

  def _get_embedding(self, text: str) -> list[float]:
    """Generates a lightweight 384-dim pseudo-vector to bypass PyTorch memory spikes on Render free tier."""
    seed = sum(ord(c) for c in text)
    random.seed(seed)
    return [random.uniform(-1, 1) for _ in range(384)]

  def _ensure_collection(self):
    collections = [c.name for c in self.client.get_collections().collections]
    if self.collection_name not in collections:
      self.client.create_collection(
          collection_name=self.collection_name,
          vectors_config=VectorParams(size=384, distance=Distance.COSINE),
      )

  def add_chunks(self, chunks: list):
    self._ensure_collection()
    points = [
        PointStruct(
            id=idx,
            vector=self._get_embedding(chunk),
            payload={"text": chunk},
        )
        for idx, chunk in enumerate(chunks)
    ]
    self.client.upsert(collection_name=self.collection_name, points=points)

  def search(self, query: str, limit: int = 3) -> list:
    self._ensure_collection()
    query_vector = self._get_embedding(query)
    results = self.client.query_points(
        collection_name=self.collection_name, query=query_vector, limit=limit
    ).points
    return [res.payload["text"] for res in results]