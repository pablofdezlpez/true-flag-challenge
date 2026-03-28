import chromadb
import numpy as np

from pathlib import Path

import cv2


class Retriever:
    def __init__(self, chroma_db_path: str | Path, k_candidates: int = 3) -> None:
        self.chroma = chromadb.PersistentClient(path=chroma_db_path)
        self._summaries = self.chroma.get_collection(name="summaries")
        self.k_candidates = k_candidates

    # Retrieval function
    def query_by_image(
        self,
        query: bytes,
    ) -> list[str]:
        """Retrieve relevant documents based on an image query.
        Query is done against image embeddings in the "images" collection"""
        query = cv2.imdecode(np.frombuffer(query, np.uint8), -1)
        summary_results = self._summaries.query(
            query_images=[query],
            n_results=self.k_candidates,  # Over-fetch to have more pool
            include=["metadatas"],
        )
        print(f"Found {len(summary_results['metadatas'][0])} summary matches")
        return summary_results["metadatas"][0]

    def query_by_text(
        self,
        query: str,
    ) -> list[str]:
        """Retrieve relevant documents based on a text query.
        Query is done hierarchical way: first against summary, then against text chunks of matched articles
        """

        summary_results = self._summaries.query(
            query_texts=[query],
            n_results=self.k_candidates,  # Over-fetch to have more pool
            include=["metadatas"],
        )
        print(f"Found {len(summary_results['metadatas'][0])} summary matches")
        return summary_results["metadatas"][0]


if __name__ == "__main__":  # Load data
    retriever = Retriever("chroma_db")
    # Test query
    query = "Does X cure illnesses?"
    results = retriever.query_by_text(query)
    for result in results:
        print(result)
