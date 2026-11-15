"""Image validation and PNG rendering, independent of the HTTP framework."""
from io import BytesIO
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


ALLOWED_CONTENT_TYPES = {
    "application/octet-stream",
    "image/bmp",
    "image/jpeg",
    "image/png",
    "image/tiff",
    "image/webp",
}


class ImageInputError(ValueError):
    def __init__(self, message, status_code=422):
        super().__init__(message)
        self.status_code = status_code


def read_image(file_object, content_type, max_bytes, max_pixels):
    if content_type and content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise ImageInputError("Unsupported upload media type", status_code=415)
    payload = file_object.read(max_bytes + 1)
    if not payload:
        raise ImageInputError("Uploaded file is empty")
    if len(payload) > max_bytes:
        raise ImageInputError("Uploaded file exceeds the byte limit", status_code=413)
    try:
        with Image.open(BytesIO(payload)) as opened:
            width, height = opened.size
            if width * height > max_pixels:
                raise ImageInputError(
                    "Decoded image exceeds the pixel limit", status_code=413
                )
            return ImageOps.exif_transpose(opened).convert("RGB")
    except ImageInputError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ImageInputError("Uploaded file is not a valid supported image") from error


def render_png(image, probability, output, threshold, alpha):
    binary = probability >= threshold
    if output == "mask":
        rendered = Image.fromarray((binary * 255).astype(np.uint8), mode="L")
        suffix = "vessels"
    else:
        overlay = np.asarray(image, dtype=np.float32).copy()
        overlay[binary] = (
            (1.0 - alpha) * overlay[binary]
            + alpha * np.array([255.0, 0.0, 0.0], dtype=np.float32)
        )
        rendered = Image.fromarray(np.uint8(np.clip(overlay, 0, 255)), mode="RGB")
        suffix = "overlay"
    buffer = BytesIO()
    rendered.save(buffer, format="PNG")
    return buffer.getvalue(), suffix


def safe_download_stem(filename):
    stem = Path(filename or "fundus").stem
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", stem).strip("._")
    return (cleaned or "fundus")[:64]
