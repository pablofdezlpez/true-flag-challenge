import chromadb

from pathlib import Path

from src.embeddings import (
    embed_texts,
    embed_image_bytes,
)
from src.config import minimum_similarity


class Retriever:
    def __init__(self, chroma_db_path: str | Path, k_candidates: int = 3) -> None:
        self.chroma = chromadb.PersistentClient(path=chroma_db_path)
        self._summaries = self.chroma.get_collection(name="summaries")
        self._text_chunks = self.chroma.get_collection(name="text_chunks")
        self._images = self.chroma.get_collection(name="images")
        self.k_candidates = k_candidates

    # Retrieval function
    def query_by_image(
        self,
        query: bytes,
    ) -> list[str]:
        """Retrieve relevant documents based on an image query.
        Query is done against image embeddings in the "images" collection"""
        query_embedding = embed_image_bytes(query)

        results = self._images.query(
            query_embeddings=[query_embedding],
            n_results=self.k_candidates,
            include=["metadatas", "distances"],
        )

        return self.parse_results(results)

    def query_by_text(
        self,
        query: str,
    ) -> list[str]:
        """Retrieve relevant documents based on a text query.
        Query is done hierarchical way: first against summary, then against text chunks of matched articles
        """
        query_embedding = embed_texts([query])[0]

        summary_results = self._summaries.query(
            query_embeddings=[query_embedding],
            n_results=self.k_candidates * 5,  # Over-fetch to have more pool
            include=["metadatas", "distances"],
        )
        print(f"Found {len(summary_results['metadatas'][0])} summary matches")
        matched_article_ids = [
            metadata["article_id"] for metadata in summary_results["metadatas"][0]
        ]
        if not matched_article_ids:
            return []

        chunk_results = self._text_chunks.query(
            query_embeddings=[query_embedding],
            n_results=max(self.k_candidates * 10, self.k_candidates),
            where={"article_id": {"$in": matched_article_ids}},
            include=["metadatas", "distances"],
        )

        print(f"Found {len(chunk_results['metadatas'][0])} chunk matches")
        return self.parse_results_unique_article(chunk_results, self.k_candidates)

    @staticmethod
    def parse_results(results: dict) -> list[str]:
        retrieved_docs = []
        for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
            if distance >= minimum_similarity:
                document = f"URL: {metadata['url']}\nTitle: {metadata['title']}\n\nFull Text: {metadata['text']}"
                retrieved_docs.append(document)

        print(
            f"Returning {len(retrieved_docs)} retrieved documents after filtering with minimum similarity"
        )
        return retrieved_docs

    @staticmethod
    def parse_results_unique_article(results: dict, k_candidates: int) -> list[str]:
        """Return up to k documents, keeping only the best chunk per article_id."""
        retrieved_docs = []
        seen_article_ids = set()

        for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
            if distance < minimum_similarity:
                continue

            article_id = metadata.get("article_id")
            if article_id in seen_article_ids:
                continue

            seen_article_ids.add(article_id)
            document = f"URL: {metadata['url']}\nTitle: {metadata['title']}\n\nFull Text: {metadata['text']}"
            retrieved_docs.append(document)

            if len(retrieved_docs) >= k_candidates:
                break

        print(
            f"Returning {len(retrieved_docs)} retrieved documents with distinct article_id"
        )
        return retrieved_docs


if __name__ == "__main__":  # Load data
    retriever = Retriever("chroma_db")
    # Test query
    query = "Does X cure illnesses?"
    results = retriever.query_by_text(query)
    for result in results:
        print(result)
