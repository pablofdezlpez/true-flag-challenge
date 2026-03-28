import argparse
import csv
import logging
import chromadb
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

from src.config import chunk_size, chunk_overlap

logger = logging.getLogger(__name__)

CHROMA_BATCH_LIMIT = 5_000
MAX_METADATA_TEXT_LEN = 5_000  # ChromaDB ~32 KB metadata limit

# Special characters required for non latin languages, as well as zero-width space to prevent empty chunks after splitting
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=[
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "\u200b",
        "\uff0c",
        "\u3001",
        "\uff0e",
        "\u3002",
        "",
    ],
)


class Indexer:
    """Ingests a CSV of articles into three ChromaDB collections: summaries, text_chunk and images
    images are taken from cr_images or defaulted to meta_images
    """

    # TODO: Late realization of improvement, I should have planned for batch indexing of the documents to accelerate the process.

    def __init__(self, chroma_db_path: str | Path) -> None:
        db_path = Path(chroma_db_path)
        db_path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(db_path))
        self._summaries = self._client.get_or_create_collection(
            name="summaries", embedding_function=OpenCLIPEmbeddingFunction()
        )

    def index_csv(self, csv_path: str | Path) -> None:
        """Read a CSV and index all articles into ChromaDB."""
        with open(csv_path, "r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            batch_size = 500
            batch = {"article_id": [], "metadata": [], "documents": []}
            for idx, row in enumerate(reader):
                article_id = f"article_{idx}"
                title = row["title"]
                summary = row["summary"]
                text = row["text"]
                cr_image = row["cr_image"]
                meta_image = row["meta_image"]
                url = row["url"]
                batch['article_id'].append(article_id)
                batch['metadata'].append({
                    "title": title,
                    "summary": summary,
                    "text": text,
                    "cr_image": cr_image,
                    "meta_image": meta_image,
                    "url": url,
                })
                batch['documents'].append(summary)
                if len(batch['article_id']) >= batch_size:
                    self._summaries.upsert(
                        ids=batch['article_id'],
                        documents=batch['documents'],
                        metadatas=batch['metadata'],
                    )
                    batch = {"article_id": [], "metadata": [], "documents": []}
                print("[%s/%s] Indexing: %s", idx + 1, title[:80])
            else: # If any remaining articles in batch after loop, index them as well
                if batch['article_id']:
                    self._summaries.upsert(
                        ids=batch['article_id'],
                        documents=batch['documents'],
                        metadatas=batch['metadata'],
                    )


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
    indexer = Indexer(chroma_db_path=args.chroma_db_path)
    indexer.index_csv(
        args.csv_path,
    )
