import numpy as np
import io

from google import genai
from google.genai import types
from PIL import Image
from urllib.request import urlopen, Request

from src.config import EMBEDDING_MODEL

# vision encoder
EMBEDDING_CLIENT = genai.Client()


def _is_valid_image_bytes(data) -> bool:
    """Return True if bytes can be parsed as an image by Pillow."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
        return True
    except Exception:
        return False


def download_image(image_url: str):
    """Download an image from URL and return raw bytes."""
    try:
        request = Request(
            image_url,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urlopen(request) as response:
            content_type = (response.headers.get("Content-Type") or "").lower()
            if content_type and not content_type.startswith("image/"):
                return None

            image_bytes = response.read()
            if not image_bytes:
                return None

            return image_bytes if _is_valid_image_bytes(image_bytes) else None
    except Exception:
        return None


def embed_texts(text: list[str]) -> np.ndarray:
    """Embed text using a pretrained text model."""
    result = EMBEDDING_CLIENT.models.embed_content(model=EMBEDDING_MODEL, contents=text)
    embeddings = result.embeddings
    return [np.array(embedding.values) for embedding in embeddings]


def embed_image_bytes(image_bytes: bytes) -> np.ndarray | None:
    """
    Embed a raw image provided as bytes.

    The bytes are converted to a base64 data-URI
    """
    result = EMBEDDING_CLIENT.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")],
    )
    return np.array(result.embeddings[0].values)


def embed_image_url(image_url: str) -> np.ndarray | None:
    """
    Embed an image from a URL.

    The image is downloaded and converted to bytes, then embedded."""
    image_bytes = download_image(image_url)
    if not image_bytes:
        return None
    if not _is_valid_image_bytes(image_bytes):
        return None

    return embed_image_bytes(image_bytes)
