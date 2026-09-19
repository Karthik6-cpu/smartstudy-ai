"""
Automated Test Suite for SmartStudy AI RAG Pipeline.
Tests PDF extraction, chunking, embeddings, FAISS indexing, persistence,
retrieval, citation generation, document deletion, and error handling.
"""

import sys
import os
import io
import shutil
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pypdf
from pypdf import PdfWriter

from rag.pdf_processor import (
    PDFProcessor,
    clean_text,
    chunk_text,
    EmptyPDFError,
    CorruptedPDFError,
    NoExtractableTextError,
)
from rag.embeddings import EmbeddingService
from rag.vector_store import VectorStore
from rag.retriever import RAGRetriever


def create_sample_cmmi_pdf() -> bytes:
    """Create a multi-page synthetic PDF containing MCA Software Engineering notes on CMMI."""
    writer = PdfWriter()

    # Page 1: CMMI Overview
    writer.add_blank_page(width=612, height=792)
    # Note: pypdf can write text into pages using page annotations or we can use ContentStream/text annotations
    # For a deterministic synthetic PDF that pypdf can read text from, let's create it via stream or simple text
    # Let's inspect: In pypdf >= 4.0, we can add text or annotations, or write a raw minimal PDF with text stream.
    return create_minimal_text_pdf([
        "Capability Maturity Model Integration (CMMI) is a process level improvement training and appraisal program.\n"
        "Developed by Carnegie Mellon University, it is required by many United States Department of Defense and U.S. Government contracts.\n"
        "CMMI can be used to guide process improvement across a project, a division, or an entire organization.",

        "CMMI defines five maturity levels for processes:\n"
        "1. Initial: Processes are unpredictable, poorly controlled and reactive.\n"
        "2. Managed: Processes are planned, performed, measured and controlled.\n"
        "3. Defined: Processes are characterized for the organization and are proactive.\n"
        "4. Quantitatively Managed: Processes are controlled using statistical techniques.\n"
        "5. Optimizing: Focus is on continual process improvement and innovation."
    ])


def create_minimal_text_pdf(pages_text: list[str]) -> bytes:
    """Construct a compliant minimal PDF byte stream containing text on each page."""
    # Build standard PDF object stream
    objects = []
    page_obj_ids = []

    # Obj 1: Catalog (placeholder)
    # Obj 2: Outlines
    # Obj 3: Pages
    # For each page: Page Obj, Font Obj, Contents Obj

    current_id = 4
    font_id = current_id
    current_id += 1

    content_ids = []
    page_ids = []

    for text in pages_text:
        p_id = current_id
        current_id += 1
        page_ids.append(p_id)

        c_id = current_id
        current_id += 1
        content_ids.append(c_id)

    # Catalog & Pages
    pdf_parts = []
    offsets = {}

    def add_obj(obj_id: int, content: str):
        offsets[obj_id] = sum(len(p.encode("latin1")) for p in pdf_parts)
        pdf_parts.append(f"{obj_id} 0 obj\n{content}\nendobj\n")

    pdf_parts.append("%PDF-1.4\n")

    # Catalog
    add_obj(1, "<< /Type /Catalog /Pages 2 0 R >>")

    # Pages tree
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    add_obj(2, f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>")

    # Standard Font (Helvetica)
    add_obj(font_id, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    # Add each page and its content stream
    for idx, (p_id, c_id, text) in enumerate(zip(page_ids, content_ids, pages_text)):
        # Format text in PDF text operators
        lines = text.split("\n")
        stream_cmds = ["BT", "/F1 12 Tf", "50 720 Td", "14 TL"]
        for line in lines:
            escaped_line = line.replace("(", "\\(").replace(")", "\\)")
            stream_cmds.append(f"({escaped_line}) '")
        stream_cmds.append("ET")
        stream_content = "\n".join(stream_cmds)

        stream_bytes_len = len(stream_content.encode("latin1"))
        add_obj(c_id, f"<< /Length {stream_bytes_len} >>\nstream\n{stream_content}\nendstream")

        add_obj(
            p_id,
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {c_id} 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> >>"
        )

    # Cross-reference table
    xref_offset = sum(len(p.encode("latin1")) for p in pdf_parts)
    num_objs = current_id
    xref_table = [f"xref\n0 {num_objs}\n0000000000 65535 f \n"]
    for i in range(1, num_objs):
        offset = offsets.get(i, 0)
        xref_table.append(f"{offset:010d} 00000 n \n")

    trailer = (
        f"trailer\n<< /Size {num_objs} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    )

    pdf_string = "".join(pdf_parts) + "".join(xref_table) + trailer
    return pdf_string.encode("latin1")


def test_pdf_extraction_and_chunking():
    """Verify PDFProcessor extracts text, page numbers, and chunks correctly."""
    print("Testing PDFProcessor text extraction and chunking...")
    pdf_bytes = create_sample_cmmi_pdf()
    processor = PDFProcessor()

    chunks, stats = processor.process_pdf(pdf_bytes, "se_cmmi_notes.pdf")

    assert stats["total_pages"] == 2, f"Expected 2 pages, got {stats['total_pages']}"
    assert stats["total_chunks"] >= 2, f"Expected at least 2 chunks, got {stats['total_chunks']}"
    assert "CMMI" in chunks[0]["text"]
    assert chunks[0]["page"] == 1
    assert chunks[-1]["page"] == 2
    assert "Optimizing" in chunks[-1]["text"]
    print(" PDF extraction and chunking passed.")


def test_pdf_error_handling():
    """Verify error handling on empty, corrupted, and zero-text PDFs."""
    print("Testing PDF error handling...")
    processor = PDFProcessor()

    # 1. Empty PDF bytes
    try:
        processor.process_pdf(b"", "empty.pdf")
        assert False, "Should have raised EmptyPDFError"
    except EmptyPDFError:
        pass

    # 2. Corrupted PDF bytes
    try:
        processor.process_pdf(b"Not a real pdf at all!", "corrupted.pdf")
        assert False, "Should have raised CorruptedPDFError"
    except CorruptedPDFError:
        pass

    # 3. PDF with blank page (no text)
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    stream = io.BytesIO()
    writer.write(stream)
    blank_bytes = stream.getvalue()

    try:
        processor.process_pdf(blank_bytes, "blank.pdf")
        assert False, "Should have raised NoExtractableTextError"
    except NoExtractableTextError:
        pass

    print(" PDF error handling passed.")


def test_vector_store_persistence_and_retrieval():
    """Verify FAISS vector store indexing, persistence to disk, reload, and retrieval."""
    print("Testing VectorStore persistence and retrieval...")
    temp_dir = Path(tempfile.mkdtemp())

    try:
        pdf_bytes = create_sample_cmmi_pdf()
        processor = PDFProcessor(storage_dir=temp_dir / "documents")
        chunks, doc_stats = processor.process_pdf(pdf_bytes, "cmmi_guide.pdf")

        embedding_service = EmbeddingService.get_instance()
        chunk_texts = [c["text"] for c in chunks]
        embeddings = embedding_service.embed_texts(chunk_texts)

        # 1. Initialize store and add document
        store = VectorStore(vector_dir=temp_dir / "vector_store")
        store.add_document("cmmi_guide.pdf", chunks, embeddings, doc_stats)

        assert store.index.ntotal == len(chunks)
        assert len(store.chunks) == len(chunks)
        assert "cmmi_guide.pdf" in store.documents

        # 2. Test persistence: create a new VectorStore instance from same directory
        store2 = VectorStore(vector_dir=temp_dir / "vector_store")
        assert store2.index.ntotal == len(chunks), "Persisted index should retain vectors"
        assert len(store2.chunks) == len(chunks), "Persisted metadata should retain chunks"
        assert "cmmi_guide.pdf" in store2.documents

        # 3. Test Retrieval with RAGRetriever
        retriever = RAGRetriever(embedding_service, store2)
        query = "Explain the 5 maturity levels of CMMI"
        results = retriever.retrieve(query, top_k=2)

        assert len(results) > 0, "Expected search results"
        top_match = results[0]
        assert "Initial" in top_match["text"] or "Maturity" in top_match["text"]
        assert top_match["filename"] == "cmmi_guide.pdf"
        assert top_match["score"] > 0.3

        # 4. Test RAG prompt construction & citations
        prompt = retriever.build_rag_prompt(query, results)
        assert "Context from Uploaded Notes:" in prompt
        assert "cmmi_guide.pdf" in prompt
        assert "Student Question:" in prompt

        citations = retriever.format_citations(results)
        assert len(citations) == len(results)
        assert citations[0]["filename"] == "cmmi_guide.pdf"
        assert citations[0]["score"] > 0.0

        # 5. Test Document Deletion
        deleted = store2.delete_document("cmmi_guide.pdf")
        assert deleted is True
        assert store2.index.ntotal == 0
        assert len(store2.chunks) == 0
        assert "cmmi_guide.pdf" not in store2.documents

        # Reload after delete to confirm deletion persisted
        store3 = VectorStore(vector_dir=temp_dir / "vector_store")
        assert store3.index.ntotal == 0
        assert len(store3.chunks) == 0

        print(" VectorStore persistence, retrieval, and deletion passed.")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_pdf_extraction_and_chunking()
    test_pdf_error_handling()
    test_vector_store_persistence_and_retrieval()
    print("\nAll RAG pipeline tests executed successfully! 🎉")
