from src.embeddings import (
    embed_image_bytes,
    embed_image_url,
    embed_texts,
)
import numpy as np


def test_text_embedding():
    query = "What is the capital of France?"
    embedding = embed_texts([query])[0]
    assert isinstance(embedding, np.ndarray)
    assert len(embedding) > 0


def test_image_bytes_embedding():
    image_path = "test/test.jpg"
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    embedding = embed_image_bytes(image_bytes)
    assert isinstance(embedding, np.ndarray)
    assert len(embedding) > 0


def test_image_url_embedding():
    image_url = "https://static.vecteezy.com/system/resources/thumbnails/057/068/323/small/single-fresh-red-strawberry-on-table-green-background-food-fruit-sweet-macro-juicy-plant-image-photo.jpg"
    embedding = embed_image_url(image_url)
    assert isinstance(embedding, np.ndarray)
    assert len(embedding) > 0


def test_image_url_embedding_with_invalid_url_returns_none():
    image_url = "https://example.com/invalid_image.jpg"
    embedding = embed_image_url(image_url)
    assert embedding is None
