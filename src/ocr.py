import base64
import logging
from pathlib import Path

from mistralai import Mistral

from src.settings import settings

logger = logging.getLogger(__name__)


# TODO def check_page_limit()-> Literal[True]


def _get_mime_type(filename: str) -> str:
    """
    Maps file extensions to appropriate MIME types for Mistral OCR.

    Args:
        filename: The name of the file including extension.

    Returns:
        The appropriate MIME type string.
    """
    ext = Path(filename).suffix.lower()
    mime_types = {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
    }
    return mime_types.get(ext, "application/octet-stream")


# @traceable(name="mistral_ocr_processing") #TODO : re-implement traceable with Attachement
def uncached_ocr_document_with_mistral(file_content: bytes, filename: str) -> str:
    """
    Processes a document using Mistral OCR and returns its content as Markdown.

    Args:
        file_content: The byte content of the file.
        filename: The name of the file.

    Returns:
        The extracted Markdown content as a string.

    Raises:
        ValueError: If the Mistral API key is not set.
        MistralException: For API-related errors (e.g., file too large, page limit).
    """
    client = Mistral(api_key=settings.get_mistral_api_key_or_raise())

    logger.info(f"Processing document '{filename}' with Mistral OCR...")

    # Encode file content to base64
    b64_content = base64.b64encode(file_content).decode("utf-8")

    # Get the appropriate MIME type based on file extension
    mime_type = _get_mime_type(filename)
    data_uri = f"data:{mime_type};base64,{b64_content}"

    # Process with Mistral OCR
    # Note: Mistral's OCR API doesn't have a direct page limit param.
    # We rely on their default limits or would need to pre-process PDFs to enforce our custom limit.
    # For now, we'll let the API handle it and catch potential errors.
    resp = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": data_uri,
        },
    )

    if not resp.pages:
        logger.warning(f"Mistral OCR returned no pages for file '{filename}'.")
        return ""  # TODO : raise or retur error instead of failing silently

    if len(resp.pages) > settings.max_pages_per_file:
        logger.warning(
            f"File '{filename}' has {len(resp.pages)} pages, which exceeds the limit of {settings.max_pages_per_file}. Truncating."
        )
        # We process only up to the page limit
        pages = resp.pages[: settings.max_pages_per_file]
    else:
        pages = resp.pages

    # Combine markdown from all pages
    full_markdown = "\n\n---\n\n".join(page.markdown for page in pages)
    logger.info(f"Successfully processed {len(pages)} pages from '{filename}'.")
    logger.debug(f"{full_markdown[:100]=}")
    return full_markdown
