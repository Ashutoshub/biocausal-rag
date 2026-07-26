from neo4j import GraphDatabase

class KnowledgeGraphEngine:
    def __init__(self, uri="bolt://localhost:7687", auth=("neo4j", "password123")):
        self.driver = GraphDatabase.driver(uri, auth=auth)

    def close(self):
        self.driver.close()

    def insert_triplet(self, subject: str, relation: str, object_node: str):
        """Creates nodes and dynamic causal relationship in Neo4j graph."""
        query = (
            f"MERGE (a:Entity {{name: $subject}}) "
            f"MERGE (b:Entity {{name: $object_node}}) "
            f"MERGE (a)-[r:{relation}]->(b) "
            f"RETURN a, r, b"
        )
        with self.driver.session() as session:
            session.run(query, subject=subject, object_node=object_node)

    def k_hop_subgraph(self, start_entity: str, k: int = 2) -> list:
        """Retrieves k-hop causal path traversal context from the graph."""
        query = (
            f"MATCH path = (a:Entity {{name: $entity}})-[*1..{k}]-(b:Entity) "
            f"RETURN path LIMIT 20"
        )
        with self.driver.session() as session:
            result = session.run(query, entity=start_entity.upper())
            paths = []
            for record in result:
                path = record["path"]
                nodes = [node["name"] for node in path.nodes]
                paths.append(" -> ".join(nodes))
            return paths