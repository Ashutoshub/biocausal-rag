import importlib
import unittest
from unittest.mock import patch


class KnowledgeGraphEngineTests(unittest.TestCase):
    def test_module_imports(self):
        module = importlib.import_module("src.graph_db")
        self.assertTrue(hasattr(module, "KnowledgeGraphEngine"))

    def test_init_without_driver_dependency_raises_helpful_error(self):
        module = importlib.import_module("src.graph_db")
        with patch.object(module, "GraphDatabase", None):
            with self.assertRaisesRegex(ImportError, "pip install neo4j"):
                module.KnowledgeGraphEngine()


if __name__ == "__main__":
    unittest.main()
