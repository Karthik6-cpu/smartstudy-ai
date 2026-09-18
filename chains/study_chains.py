"""
LangChain LCEL Chains for SmartStudy AI.
Modular Runnable pipelines connecting ChatPromptTemplates, LangChain Retrievers,
ChatOllama, and StrOutputParser with token streaming.
"""

from typing import Generator, List, Dict, Any, Tuple, Optional
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document

from prompts.templates import (
    GENERAL_STUDY_PROMPT,
    RAG_STUDY_PROMPT,
    QUIZ_GENERATION_PROMPT,
    STUDY_PLANNER_PROMPT,
)
from llm.langchain_client import get_chat_ollama, LangChainOllamaFactory
from rag.retriever import RAGRetriever
from config.settings import DEFAULT_MODEL, DEFAULT_OLLAMA_HOST


def create_general_study_chain(
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_OLLAMA_HOST,
    temperature: float = 0.3,
):
    """
    Construct LCEL chain for standard study assistant conversation.
    Pipeline: GENERAL_STUDY_PROMPT | ChatOllama | StrOutputParser
    """
    llm = get_chat_ollama(model=model, base_url=base_url, temperature=temperature)
    return GENERAL_STUDY_PROMPT | llm | StrOutputParser()


def create_rag_chain(
    retriever: Optional[RAGRetriever] = None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_OLLAMA_HOST,
    temperature: float = 0.2,
):
    """
    Construct LCEL chain for retrieval-augmented generation.
    Pipeline: { context: retriever, question: RunnablePassthrough } | RAG_STUDY_PROMPT | ChatOllama | StrOutputParser
    """
    active_retriever = retriever or RAGRetriever()
    llm = get_chat_ollama(model=model, base_url=base_url, temperature=temperature)

    def retrieve_and_format(input_dict: Dict[str, Any]) -> str:
        q = input_dict.get("question", "")
        docs = active_retriever.retrieve_documents(q)
        return active_retriever.format_docs_for_context(docs)

    rag_chain = (
        RunnablePassthrough.assign(context=RunnableLambda(retrieve_and_format))
        | RAG_STUDY_PROMPT
        | llm
        | StrOutputParser()
    )
    return rag_chain


def stream_rag_chat(
    question: str,
    retriever: RAGRetriever,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_OLLAMA_HOST,
    temperature: float = 0.2,
) -> Tuple[Generator[str, None, None], List[Document]]:
    """
    Stream RAG response while capturing the exact LangChain Document citations.

    Returns:
        Tuple of (stream_generator, retrieved_documents_list)
    """
    # Verify connection first
    is_ok, msg = LangChainOllamaFactory.validate_connection(base_url)
    if not is_ok:
        def err_gen():
            yield f"⚠️ **Ollama Service Unavailable:**\n\n{msg}"
        return err_gen(), []

    # 1. Retrieve documents using LangChain BaseRetriever
    docs = retriever.retrieve_documents(question)
    context_str = retriever.format_docs_for_context(docs)

    # 2. Build prompt and stream tokens via LCEL
    try:
        llm = get_chat_ollama(model=model, base_url=base_url, temperature=temperature)
        chain = RAG_STUDY_PROMPT | llm | StrOutputParser()

        def response_stream():
            try:
                for chunk in chain.stream({"context": context_str, "question": question}):
                    yield chunk
            except Exception as e:
                yield f"\n\n❌ **LangChain Stream Error:** `{str(e)}`"

        return response_stream(), docs

    except Exception as e:
        def exc_gen():
            yield f"❌ **Error initializing LangChain Ollama model:** `{str(e)}`"
        return exc_gen(), docs


def create_quiz_chain(
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_OLLAMA_HOST,
    temperature: float = 0.4,
):
    """
    Construct LCEL chain for generating mock quizzes and practice questions.
    Pipeline: QUIZ_GENERATION_PROMPT | ChatOllama | StrOutputParser
    """
    llm = get_chat_ollama(model=model, base_url=base_url, temperature=temperature)
    return QUIZ_GENERATION_PROMPT | llm | StrOutputParser()


def create_planner_chain(
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_OLLAMA_HOST,
    temperature: float = 0.3,
):
    """
    Construct LCEL chain for semester timetable and revision planning.
    Pipeline: STUDY_PLANNER_PROMPT | ChatOllama | StrOutputParser
    """
    llm = get_chat_ollama(model=model, base_url=base_url, temperature=temperature)
    return STUDY_PLANNER_PROMPT | llm | StrOutputParser()
