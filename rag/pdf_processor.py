"""
PDF Processor for SmartStudy AI.
Handles PDF reading, text cleaning, metadata extraction, SHA-256 duplicate hashing,
password protection detection, optional OCR fallback, and text chunking.
"""

import os
import re
import io
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional
import pypdf
from pypdf.errors import PdfStreamError, EmptyFileError

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, DOCUMENTS_DIR

# Optional OCR check
HAS_OCR = False
try:
    import pytesseract
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False


class PDFProcessingError(Exception):
    """Base exception for PDF processing errors."""
    pass


class CorruptedPDFError(PDFProcessingError):
    """Raised when PDF file is corrupted or cannot be parsed."""
    pass


class PasswordProtectedPDFError(PDFProcessingError):
    """Raised when PDF file is encrypted/password-protected."""
    pass


class EmptyPDFError(PDFProcessingError):
    """Raised when PDF file has 0 pages or 0 bytes."""
    pass


class NoExtractableTextError(PDFProcessingError):
    """Raised when PDF contains pages but no text can be extracted (e.g. scanned image)."""
    pass


def compute_sha256(content: bytes) -> str:
    """Compute SHA-256 hash of file byte content for duplicate detection."""
    return hashlib.sha256(content).hexdigest()


def clean_text(text: str) -> str:
    """Normalize whitespace and remove non-printable characters from extracted text."""
    if not text:
        return ""
    # Replace null bytes
    text = text.replace("\x00", "")
    # Normalize line breaks and multiple whitespace
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """
    Split text into semantically cohesive overlapping chunks.
    Attempts splits at paragraphs, then sentences, then spaces.
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        # If not at the very end, try to find a natural break point
        if end < text_length:
            # Look for paragraph break
            p_break = text.rfind("\n\n", start, end)
            if p_break != -1 and p_break > start + (chunk_size // 3):
                end = p_break + 2
            else:
                # Look for sentence break
                s_break = max(
                    text.rfind(". ", start, end),
                    text.rfind("? ", start, end),
                    text.rfind("! ", start, end),
                    text.rfind(".\n", start, end),
                )
                if s_break != -1 and s_break > start + (chunk_size // 3):
                    end = s_break + 1
                else:
                    # Look for space
                    w_break = text.rfind(" ", start, end)
                    if w_break != -1 and w_break > start + (chunk_size // 4):
                        end = w_break + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Advance start point considering overlap
        if end >= text_length:
            break
        start = max(end - chunk_overlap, start + 1)

    return chunks


class PDFProcessor:
    """Manages PDF loading, text extraction, validation, and chunking with metadata."""

    def __init__(self, storage_dir: Union[str, Path] = DOCUMENTS_DIR):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_pdf(self, file_bytes: bytes, filename: str) -> Path:
        """Save an uploaded PDF to local disk storage."""
        safe_filename = Path(filename).name
        target_path = self.storage_dir / safe_filename
        with open(target_path, "wb") as f:
            f.write(file_bytes)
        return target_path

    def process_pdf(
        self,
        file_input: Union[str, Path, bytes, io.BytesIO],
        filename: str
    ) -> Tuple[List[Dict], Dict]:
        """
        Extract text from PDF, perform chunking, and build chunk metadata.
        Includes SHA-256 calculation, password protection check, and optional OCR fallback.

        Args:
            file_input: Path to file or raw bytes/BytesIO.
            filename: Original file name.

        Returns:
            Tuple of (chunks_list, document_stats_dict)

        Raises:
            EmptyPDFError, CorruptedPDFError, PasswordProtectedPDFError, NoExtractableTextError
        """
        file_bytes = b""
        if isinstance(file_input, (str, Path)):
            path_obj = Path(file_input)
            if not path_obj.exists() or path_obj.stat().st_size == 0:
                raise EmptyPDFError(f"PDF file '{filename}' is empty or does not exist.")
            with open(path_obj, "rb") as f:
                file_bytes = f.read()
            stream = io.BytesIO(file_bytes)
        elif isinstance(file_input, bytes):
            if len(file_input) == 0:
                raise EmptyPDFError(f"PDF '{filename}' contains 0 bytes.")
            file_bytes = file_input
            stream = io.BytesIO(file_bytes)
        elif isinstance(file_input, io.BytesIO):
            file_input.seek(0)
            file_bytes = file_input.getvalue()
            if len(file_bytes) == 0:
                raise EmptyPDFError(f"PDF '{filename}' contains 0 bytes.")
            stream = file_input
        else:
            raise ValueError("Unsupported file input type")

        file_hash = compute_sha256(file_bytes)
        file_size_kb = round(len(file_bytes) / 1024, 1)

        try:
            reader = pypdf.PdfReader(stream)
            num_pages = len(reader.pages)
        except (PdfStreamError, EmptyFileError) as e:
            raise CorruptedPDFError(f"Could not read '{filename}'. The file may be damaged or corrupted: {str(e)}")
        except Exception as e:
            raise CorruptedPDFError(f"Error parsing PDF '{filename}': {str(e)}")

        # Check for password encryption
        if getattr(reader, "is_encrypted", False):
            try:
                decrypted = reader.decrypt("")
                if not decrypted:
                    raise PasswordProtectedPDFError(
                        f"PDF '{filename}' is password-protected. Please unlock the file before uploading."
                    )
            except Exception:
                raise PasswordProtectedPDFError(
                    f"PDF '{filename}' is password-protected. Please unlock the file before uploading."
                )

        if num_pages == 0:
            raise EmptyPDFError(f"PDF '{filename}' contains 0 pages.")

        all_chunks = []
        total_extracted_chars = 0
        pages_with_text = 0

        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1
            raw_text = ""
            try:
                raw_text = page.extract_text() or ""
            except Exception:
                raw_text = ""

            # Optional OCR fallback if page has 0 extractable text
            if not raw_text.strip() and HAS_OCR:
                try:
                    for img_obj in page.images:
                        img = Image.open(io.BytesIO(img_obj.data))
                        ocr_txt = pytesseract.image_to_string(img)
                        if ocr_txt.strip():
                            raw_text += "\n" + ocr_txt
                except Exception:
                    pass

            cleaned = clean_text(raw_text)
            if cleaned:
                pages_with_text += 1
                total_extracted_chars += len(cleaned)
                page_chunks = chunk_text(cleaned)

                for chunk_idx, text_segment in enumerate(page_chunks):
                    chunk_id = f"{filename}_p{page_num}_c{chunk_idx + 1}"
                    all_chunks.append({
                        "id": chunk_id,
                        "filename": filename,
                        "page": page_num,
                        "chunk_index": chunk_idx + 1,
                        "text": text_segment,
                        "char_count": len(text_segment),
                    })

        if total_extracted_chars == 0 or len(all_chunks) == 0:
            ocr_hint = (
                "The PDF appears to be a scanned document.\n"
                "Install `pytesseract` and Tesseract OCR engine for scanned image extraction."
                if not HAS_OCR else
                "The document could not be read with text or OCR extraction."
            )
            raise NoExtractableTextError(
                f"No extractable text found in '{filename}'.\n{ocr_hint}"
            )

        now_str = datetime.now().strftime("%d %b %Y")
        stats = {
            "filename": filename,
            "file_hash": file_hash,
            "file_size_kb": file_size_kb,
            "total_pages": num_pages,
            "pages_with_text": pages_with_text,
            "total_chunks": len(all_chunks),
            "total_characters": total_extracted_chars,
            "upload_date": now_str,
            "status": "Ready",
            "embedding_status": "Indexed",
            "error": None,
        }

        return all_chunks, stats
