"""
Hierarchical Map-Reduce Document Summarization Engine for SmartStudy AI.
Ensures grounded, structured summaries for any size PDF without context overflow.
Supports:
- Quick Summary (concise overview)
- Detailed Summary (comprehensive breakdown)
- Exam Notes (Definitions, Key Concepts, Important Points, Examples, Exam Revision)
- Revision Summary (High-yield bullet points)
- Chapter/Section Summary
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from rag.vector_store import VectorStore
from llm.langchain_client import get_chat_ollama, LangChainOllamaFactory
from langchain_core.messages import HumanMessage, SystemMessage
from config.settings import DEFAULT_MODEL, DEFAULT_OLLAMA_HOST


SUMMARIZER_SYSTEM_PROMPT = """You are an expert academic summarizer for Computer Science and MCA university students.
Your summaries must be STRICTLY grounded in the provided document text.
Do not invent, hallucinate, or extrapolate facts beyond what is in the document.
Preserve exact technical terminology (e.g. CMMI levels, algorithm names, protocols, schemas).
If the text does not contain enough information to cover a requested topic, state:
"The uploaded document does not provide enough information to summarize this topic."
Format your output cleanly in Markdown."""


def summarize_chunk_batch(
    chunks_text: str,
    target_topic: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_OLLAMA_HOST,
) -> str:
    """Map step: Extract key technical points from a chunk batch."""
    is_ok, _ = LangChainOllamaFactory.validate_connection(base_url)
    if not is_ok:
        # Fallback key sentence extraction if Ollama offline
        lines = [line.strip() for line in chunks_text.split("\n") if len(line.strip()) > 30]
        return "\n".join(f"- {line}" for line in lines[:5])

    llm = get_chat_ollama(model=model, base_url=base_url, temperature=0.1)
    prompt = (
        f"Extract the key facts, definitions, and technical points from the following notes excerpt.\n"
        f"Keep it concise (4-6 bullet points).\n"
    )
    if target_topic:
        prompt += f"Focus especially on information relevant to '{target_topic}'.\n"
    prompt += f"\nEXCERPT:\n{chunks_text}\n"

    try:
        resp = llm.invoke([SystemMessage(content=SUMMARIZER_SYSTEM_PROMPT), HumanMessage(content=prompt)])
        return resp.content.strip()
    except Exception:
        lines = [line.strip() for line in chunks_text.split("\n") if len(line.strip()) > 30]
        return "\n".join(f"- {line}" for line in lines[:4])


def generate_document_summary(
    filename: str,
    mode: str = "Exam Notes",
    page_range: Optional[Tuple[int, int]] = None,
    topic: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_OLLAMA_HOST,
) -> Dict[str, Any]:
    """
    Generate a grounded summary using hierarchical map-reduce.

    Returns:
        Dict containing:
          - "summary": str (Markdown formatted text)
          - "sources": str (Document name and page ranges)
          - "word_count": int
          - "page_count": int
    """
    store = VectorStore()
    if filename not in store.documents:
        # Try case-insensitive matching
        matched = [f for f in store.documents if filename.lower() in f.lower()]
        if matched:
            filename = matched[0]
        else:
            return {
                "summary": f"⚠️ Document '{filename}' is not found in the indexed library.",
                "sources": "None",
                "word_count": 0,
                "page_count": 0,
            }

    chunks = store.get_document_chunks(filename, page_range=page_range)
    if not chunks:
        return {
            "summary": "⚠️ No chunks found matching the specified document or page range.",
            "sources": filename,
            "word_count": 0,
            "page_count": 0,
        }

    # Filter by topic if specified
    if topic and topic.strip():
        t_clean = topic.strip().lower()
        topic_chunks = [c for c in chunks if t_clean in c.get("text", "").lower()]
        if len(topic_chunks) >= 2:
            chunks = topic_chunks
        elif not topic_chunks:
            return {
                "summary": f"The uploaded document does not provide enough information to summarize '{topic}'.",
                "sources": filename,
                "word_count": 0,
                "page_count": 0,
            }

    pages_covered = sorted(list(set(c.get("page", 1) for c in chunks)))
    page_str = f"{min(pages_covered)}–{max(pages_covered)}" if len(pages_covered) > 1 else str(pages_covered[0])
    source_citation = f"📄 `{filename}` (Pages: {page_str})"

    # Check total length to decide between direct or hierarchical map-reduce
    total_text = "\n\n".join(c.get("text", "") for c in chunks)

    # If document is small (< 3000 chars), synthesize directly
    if len(total_text) <= 3500:
        intermediate_notes = total_text
    else:
        # Hierarchical Map-Reduce: Batch chunks into ~2000 character windows
        batch_summaries = []
        batch_texts = []
        cur_chars = 0

        for chunk in chunks:
            c_txt = chunk.get("text", "")
            batch_texts.append(f"[Page {chunk.get('page', 1)}]: {c_txt}")
            cur_chars += len(c_txt)

            if cur_chars >= 2000:
                combined_batch = "\n".join(batch_texts)
                batch_sum = summarize_chunk_batch(combined_batch, target_topic=topic, model=model, base_url=base_url)
                batch_summaries.append(batch_sum)
                batch_texts = []
                cur_chars = 0

        if batch_texts:
            combined_batch = "\n".join(batch_texts)
            batch_sum = summarize_chunk_batch(combined_batch, target_topic=topic, model=model, base_url=base_url)
            batch_summaries.append(batch_sum)

        intermediate_notes = "\n\n".join(batch_summaries)

    # Final Synthesis (Reduce step) based on selected mode
    mode_instructions = {
        "Quick Summary": (
            "Provide a concise executive overview (150-250 words) highlighting what the document covers "
            "and its 3-5 primary takeaways."
        ),
        "Detailed Summary": (
            "Provide a comprehensive, in-depth explanation covering all sections, process steps, "
            "architectural components, and technical details from the notes."
        ),
        "Exam Notes": (
            "Format the summary strictly under these headers:\n"
            "# <Title>\n"
            "## 1. Definition\n"
            "## 2. Key Concepts\n"
            "## 3. Important Points\n"
            "## 4. Examples\n"
            "## 5. Exam Revision (Likely 5-mark and 10-mark university questions and answers)"
        ),
        "Revision Summary": (
            "Provide ultra-concise, high-yield bullet points optimized for 1-day-before-exam revision. "
            "Highlight core definitions, acronyms, and formulas only."
        ),
        "Chapter/Section Summary": (
            "Break down the material into distinct sections or chapters with subheadings and bullet points."
        ),
    }

    instructions = mode_instructions.get(mode, mode_instructions["Exam Notes"])

    final_prompt = (
        f"You are preparing a student study summary for '{filename}'.\n"
        f"Summary Mode: {mode}\n\n"
        f"INSTRUCTIONS:\n{instructions}\n\n"
        f"STRICT GROUNDING RULE: Base your summary strictly on these extracted source notes. "
        f"Do not invent external facts. Preserve technical terms accurately.\n\n"
        f"SOURCE NOTES:\n"
        f"====================================\n"
        f"{intermediate_notes}\n"
        f"====================================\n"
    )

    is_ok, _ = LangChainOllamaFactory.validate_connection(base_url)
    if is_ok:
        try:
            llm = get_chat_ollama(model=model, base_url=base_url, temperature=0.2)
            resp = llm.invoke([SystemMessage(content=SUMMARIZER_SYSTEM_PROMPT), HumanMessage(content=final_prompt)])
            final_text = resp.content.strip()
        except Exception:
            final_text = intermediate_notes
    else:
        final_text = (
            f"# Summary: {filename} ({mode})\n\n"
            f"*(Generated from local chunk extractions - Ollama currently offline)*\n\n"
            f"{intermediate_notes}"
        )

    # Append grounded citation footer
    final_output = (
        f"{final_text}\n\n"
        f"---\n"
        f"**Sources:**\n"
        f"{source_citation}\n"
    )

    word_count = len(re.findall(r"\b\w+\b", final_output))

    return {
        "summary": final_output,
        "sources": source_citation,
        "word_count": word_count,
        "page_count": len(pages_covered),
    }
