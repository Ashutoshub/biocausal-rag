import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer


class VectorSearchEngine:

  def __init__(self, collection_name="bio_papers"):
    self.collection_name = collection_name

    # 1. Connect to Qdrant Cloud if credentials exist, otherwise use embedded in-memory mode
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if qdrant_url and qdrant_api_key:
      self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    else:
      # Embedded mode keeps vectors in memory without needing a local Qdrant Docker container
      self.client = QdrantClient(":memory:")

    # 2. Defer model loading (Lazy Load) so app starts instantly without RAM spikes
    self._encoder = None

  @property
  def encoder(self):
    """Loads SentenceTransformer only when vectors are actually generated."""
    if self._encoder is None:
      self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return self._encoder

  def _ensure_collection(self):
    """Ensures collection exists before insertion or search."""
    collections = [c.name for c in self.client.get_collections().collections]
    if self.collection_name not in collections:
      self.client.create_collection(
          collection_name=self.collection_name,
          vectors_config=VectorParams(size=384, distance=Distance.COSINE),
      )

  def add_chunks(self, chunks: list):
    """Encodes text chunks and inserts vector points into Qdrant."""
    self._ensure_collection()
    points = []
    for idx, chunk in enumerate(chunks):
      vector = self.encoder.encode(chunk).tolist()
      points.append(
          PointStruct(id=idx, vector=vector, payload={"text": chunk})
      )
    self.client.upsert(collection_name=self.collection_name, points=points)

  def search(self, query: str, limit: int = 3) -> list:
    """Searches vector space for top-k semantically similar text chunks."""
    self._ensure_collection()
    query_vector = self.encoder.encode(query).tolist()
    results = self.client.query_points(
        collection_name=self.collection_name, query=query_vector, limit=limit
    ).points
    return [res.payload["text"] for res in results]