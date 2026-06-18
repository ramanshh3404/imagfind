import os
import chromadb
from typing import Dict, List, Any

class DatabaseManager:
    def __init__(self, db_dir: str = ".imagfind_db"):
        self.db_dir = os.path.abspath(db_dir)
        # Ensure database directory exists
        os.makedirs(self.db_dir, exist_ok=True)
        # Create persistent client
        self.client = chromadb.PersistentClient(path=self.db_dir)
        # Get or create collection using cosine distance metric
        self.collection = self.client.get_or_create_collection(
            name="image_search",
            metadata={"hnsw:space": "cosine"}
        )

    def add_images(self, ids: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]]):
        """Add images to the database."""
        if not ids:
            return
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def get_all(self) -> Dict[str, Any]:
        """Get all images in the database."""
        # Retrieve ids and metadatas, embeddings are not needed by default
        return self.collection.get(include=["metadatas"])

    def get_all_with_embeddings(self) -> Dict[str, Any]:
        """Get all images in the database with their embedding vectors."""
        return self.collection.get(include=["metadatas", "embeddings"])

    def delete_images(self, ids: List[str]):
        """Delete images by ID list."""
        if not ids:
            return
        self.collection.delete(ids=ids)

    def search(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar images using query embedding."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            include=["metadatas", "distances"]
        )
        
        formatted_results = []
        if results and results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i]
                # Cosine distance in Chroma is 1 - cosine_similarity.
                # So cosine_similarity = 1 - distance.
                similarity = 1.0 - distance
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "similarity": similarity,
                    "metadata": results["metadatas"][0][i]
                })
        return formatted_results

    def clear_all(self):
        """Reset the database by deleting the collection and recreating it."""
        try:
            self.client.delete_collection("image_search")
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name="image_search",
            metadata={"hnsw:space": "cosine"}
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics of the database."""
        count = self.collection.count()
        return {
            "count": count,
            "db_path": self.db_dir,
            "collection_name": "image_search"
        }
