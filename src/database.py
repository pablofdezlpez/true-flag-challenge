import argparse
import csv
import cv2
import numpy as np
import logging
import chromadb
from pathlib import Path
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from src.config import K_NEIGHBORS, CHROMA_BATCH

logger = logging.getLogger(__name__)


class Database:
    """Ingests a CSV of articles into three ChromaDB collections: summaries, text_chunk and images
    images are taken from cr_images or defaulted to meta_images
    """

    def __init__(self, chroma_db_path: str | Path) -> None:
        db_path = Path(chroma_db_path)
        db_path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(db_path))
        self._summaries = self._client.get_or_create_collection(
            name="summaries", embedding_function=OpenCLIPEmbeddingFunction()
        )
        self.k_candidates = K_NEIGHBORS

    def index_csv(self, csv_path: str | Path) -> None:
        """Read a CSV and index all articles into ChromaDB."""
        with open(csv_path, "r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            batch = {"article_id": [], "metadata": [], "documents": []}
            for idx, row in enumerate(reader):
                article_id = f"article_{idx}"
                title = row["title"]
                summary = row["summary"]
                text = row["text"]
                cr_image = row["cr_image"]
                meta_image = row["meta_image"]
                url = row["url"]
                batch["article_id"].append(article_id)
                batch["metadata"].append(
                    {
                        "title": title,
                        "summary": summary,
                        "text": text,
                        "cr_image": cr_image,
                        "meta_image": meta_image,
                        "url": url,
                    }
                )
                batch["documents"].append(summary)
                if len(batch["article_id"]) >= CHROMA_BATCH:
                    self._summaries.upsert(
                        ids=batch["article_id"],
                        documents=batch["documents"],
                        metadatas=batch["metadata"],
                    )
                    batch = {"article_id": [], "metadata": [], "documents": []}
                print("[%s/%s] Indexing: %s", idx + 1, title[:80])
            else:  # If any remaining articles in batch after loop, index them as well
                if batch["article_id"]:
                    self._summaries.upsert(
                        ids=batch["article_id"],
                        documents=batch["documents"],
                        metadatas=batch["metadata"],
                    )

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Index articles from CSV into ChromaDB"
    )
    parser.add_argument(
        "-d",
        "--csv_path",
        type=str,
        default="claim_matching_dataset.csv",
        help="Path to the CSV file containing articles",
    )
    parser.add_argument(
        "-o",
        "--chroma_db_path",
        type=str,
        default="chroma_db",
        help="Path to the ChromaDB directory for storing embeddings",
    )
    args = parser.parse_args()
    indexer = Database(chroma_db_path=args.chroma_db_path)
    indexer.index_csv(
        args.csv_path,
    )
    query = "Does X cure illnesses?"
    results = indexer.query_by_text(query)
    for result in results:
        print(result)
