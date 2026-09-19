"""
Automated Test Suite for LangChain Integration in SmartStudy AI.
Tests PromptTemplates, LangChainFAISSRetriever, LangChain Tools, and LCEL chains.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

from prompts.templates import (
    GENERAL_STUDY_PROMPT,
    RAG_STUDY_PROMPT,
    QUIZ_GENERATION_PROMPT,
    STUDY_PLANNER_PROMPT,
)
from rag.retriever import LangChainFAISSRetriever, RAGRetriever
from rag.vector_store import VectorStore
from rag.embeddings import EmbeddingService
from tools.study_tools import (
    calculate_study_schedule,
    get_mca_subject_overview,
    search_course_notes,
)
from chains.study_chains import (
    create_general_study_chain,
    create_rag_chain,
    create_quiz_chain,
    create_planner_chain,
)


def test_prompt_templates():
    """Verify all 4 prompt templates format correctly."""
    print("Testing LangChain Prompt Templates...")

    # 1. General Study Prompt
    gen_val = GENERAL_STUDY_PROMPT.format_messages(question="Explain Quicksort")
    assert len(gen_val) >= 2
    assert "SmartStudy AI" in gen_val[0].content
    assert "Explain Quicksort" in gen_val[-1].content

    # 2. RAG Study Prompt
    rag_val = RAG_STUDY_PROMPT.format_messages(
        context="CMMI defines 5 levels.",
        question="What is CMMI?"
    )
    assert len(rag_val) == 2
    assert "Context from Uploaded Notes:" in rag_val[1].content
    assert "CMMI defines 5 levels." in rag_val[1].content

    # 3. Quiz Prompt
    quiz_val = QUIZ_GENERATION_PROMPT.format_messages(
        topic="Operating Systems: Deadlocks",
        difficulty="Intermediate",
        num_questions="5"
    )
    assert len(quiz_val) == 2
    assert "Deadlocks" in quiz_val[1].content
    assert "Multiple Choice Questions" in quiz_val[1].content

    # 4. Planner Prompt
    planner_val = STUDY_PLANNER_PROMPT.format_messages(
        subjects="DSA, DBMS",
        timeline="4 Weeks",
        daily_hours="3.0",
        goal="Ace End-Semester Finals"
    )
    assert len(planner_val) == 2
    assert "DSA, DBMS" in planner_val[1].content

    print(" Prompt Templates validated successfully.")


def test_langchain_retriever_interface():
    """Verify LangChainFAISSRetriever adheres to BaseRetriever interface and produces Documents."""
    print("Testing LangChainFAISSRetriever...")
    store = VectorStore()
    retriever_wrapper = RAGRetriever(vector_store=store)

    assert isinstance(retriever_wrapper.langchain_retriever, BaseRetriever)

    # Invoke search on empty or populated store
    docs = retriever_wrapper.retrieve_documents("Test query")
    assert isinstance(docs, list)
    for doc in docs:
        assert isinstance(doc, Document)
        assert "source" in doc.metadata
        assert "page" in doc.metadata

    print(" LangChainFAISSRetriever validated successfully.")


def test_study_tools():
    """Verify LangChain study tools execute and return formatted results."""
    print("Testing LangChain Study Tools...")

    # Test schedule calculator
    calc_result = calculate_study_schedule.invoke({
        "subjects": "DSA, OS, DBMS",
        "total_weeks": 3,
        "daily_hours": 4.0,
    })
    assert "Total Available Study Hours" in calc_result
    assert "84.0 hours" in calc_result
    assert "DSA, OS, DBMS" in calc_result

    # Test curriculum lookup
    dsa_overview = get_mca_subject_overview.invoke({"subject_name": "dsa"})
    assert "Data Structures & Algorithms" in dsa_overview
    assert "Trees" in dsa_overview

    os_overview = get_mca_subject_overview.invoke({"subject_name": "os"})
    assert "Operating Systems" in os_overview
    assert "Deadlocks" in os_overview

    print(" LangChain Tools validated successfully.")


def test_lcel_chains_construction():
    """Verify LCEL chains instantiate without syntax or wiring errors."""
    print("Testing LCEL chains construction...")

    # Instantiating chains
    general_chain = create_general_study_chain()
    assert general_chain is not None

    rag_chain = create_rag_chain()
    assert rag_chain is not None

    quiz_chain = create_quiz_chain()
    assert quiz_chain is not None

    planner_chain = create_planner_chain()
    assert planner_chain is not None

    print(" LCEL chains constructed successfully.")


if __name__ == "__main__":
    test_prompt_templates()
    test_langchain_retriever_interface()
    test_study_tools()
    test_lcel_chains_construction()
    print("\nAll LangChain integration tests passed! [SUCCESS]")
