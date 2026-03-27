import argparse
import csv
import logging
import chromadb
from google.genai.errors import ClientError
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.embeddings import embed_texts, embed_image_url
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
        self._summaries = self._client.get_or_create_collection("summaries")
        self._text_chunks = self._client.get_or_create_collection("text_chunks")
        self._images = self._client.get_or_create_collection("images")

    def index_csv(self, csv_path: str | Path) -> None:
        """Read a CSV and index all articles into ChromaDB."""
        with open(csv_path, "r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)

            for idx, row in enumerate(reader):
                article_id = f"article_{idx}"
                title = row["title"]
                summary = row["summary"]
                text = row["text"]
                cr_image = row["cr_image"]
                meta_image = row["meta_image"]
                url = row["url"]

                logger.info("[%s/%s] Indexing: %s", idx + 1, title[:80])
                if self._is_article_indexed(article_id):
                    logger.info("Already indexed – skipping")
                    continue
                image_url = self._index_image(
                    article_id, title, text, url, cr_image, meta_image
                )
                self._index_summary(article_id, title, summary, url, image_url)
                self._index_text_chunks(article_id, title, text, url, image_url)

    def _index_summary(
        self,
        article_id: str,
        title: str,
        summary: str,
        url: str,
        image_url: str,
    ) -> None:
        """Create index of summaries"""
        try:
            embedding = embed_texts(summary)[0]
        except ClientError as e:
            logger.error("Failed to embed summary for %s: %s", article_id, str(e))
            return
        self._summaries.upsert(
            ids=article_id,
            embeddings=embedding,
            metadatas={
                "article_id": article_id,
                "title": title,
                "url": url,
                "image_url": image_url,
                "summary": summary,
            },
        )

    def _index_text_chunks(
        self,
        article_id: str,
        title: str,
        text: str,
        url: str,
        image_url: str,
    ) -> None:
        """Create index of text chunks."""
        chunks = TEXT_SPLITTER.split_text(text)

        ids: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict] = []
        documents: list[str] = []

        # Embed in batches to avoid oversized API calls
        for batch_start in range(0, len(chunks), CHROMA_BATCH_LIMIT):
            batch = chunks[batch_start : batch_start + CHROMA_BATCH_LIMIT]
            batch_embeddings = embed_texts(batch)

            for i, (chunk, emb) in enumerate(
                zip(batch, batch_embeddings), start=batch_start
            ):
                chunk_id = f"{article_id}_chunk_{i}"
                ids.append(chunk_id)
                embeddings.append(emb)
                metadatas.append(
                    {
                        "article_id": article_id,
                        "chunk_index": i,
                        "title": title,
                        "url": url,
                        "image_url": image_url,
                        "text": text,
                    }
                )
                documents.append(chunk)

        # Upsert in batches for ChromaDB
        for batch_start in range(0, len(ids), CHROMA_BATCH_LIMIT):
            end = batch_start + CHROMA_BATCH_LIMIT
            self._text_chunks.upsert(
                ids=ids[batch_start:end],
                embeddings=embeddings[batch_start:end],
                metadatas=metadatas[batch_start:end],
                documents=documents[batch_start:end],
            )

    def _index_image(
        self,
        article_id: str,
        title: str,
        text: str,
        url: str,
        cr_image: str,
        meta_image: str,
    ) -> str | None:
        """Create index of images. Image is taken from cr_image or meta_image if cr_image is not valid."""
        if not cr_image.strip() and not meta_image.strip():
            logger.warning("No image URL for %s – skipping image index", article_id)
            return
        try:
            embedding = embed_image_url(cr_image)
        except ClientError:
            return
        image_url = cr_image
        if embedding is None:
            logger.warning(
                "Failed to embed image for %s (URL: %s) – attempting with meta image",
                article_id,
                cr_image,
            )
            try:
                embedding = embed_image_url(meta_image)
            except ClientError:
                return
            image_url = meta_image
            if embedding is None:
                logger.error(
                    "Failed to embed meta image for %s (URL: %s) – skipping image index",
                    article_id,
                    meta_image,
                )
                return

        self._images.upsert(
            ids=article_id,
            embeddings=embedding,
            metadatas={
                "article_id": article_id,
                "title": title,
                "url": url,
                "image_url": image_url,
                "text": text,
            },
        )
        return image_url

    def _is_article_indexed(self, article_id: str) -> bool:
        """Return True when summary and at least one text chunk already exist to prevent restarting from scratch."""
        summary_record = self._summaries.get(ids=[article_id], include=[])
        has_summary = len(summary_record.get("ids", [])) > 0

        chunk_record = self._text_chunks.get(
            where={"article_id": article_id},
            limit=1,
            include=[],
        )
        has_chunk = len(chunk_record.get("ids", [])) > 0
        return has_summary and has_chunk


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
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume mode and re-index every article.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop ingestion on first error instead of continuing.",
    )
    args = parser.parse_args()
    indexer = Indexer(chroma_db_path=args.chroma_db_path)
    indexer.index_csv(
        args.csv_path,
        resume=not args.no_resume,
        continue_on_error=not args.fail_fast,
    )
