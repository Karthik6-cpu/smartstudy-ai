"""
Configuration settings for SmartStudy AI.
Centralizes app parameters, default models, navigation pages, RAG, memory, and LangGraph checkpoints.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "index.faiss"
METADATA_PATH = VECTOR_STORE_DIR / "metadata.json"
MEMORY_DIR = DATA_DIR / "memory"
MEMORY_DB_PATH = MEMORY_DIR / "student_memory.db"
GRAPH_CHECKPOINTS_PATH = MEMORY_DIR / "graph_checkpoints.db"

# Application Metadata
APP_NAME = "SmartStudy AI"
APP_SUBTITLE = "Local AI Study Assistant for MCA Students"
APP_VERSION = "3.0.0 (LangGraph Agent Workflow)"

# Ollama Defaults
DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3.2:1b")

# RAG & Embedding Settings
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 600       # Target characters per chunk
CHUNK_OVERLAP = 100    # Overlap characters to preserve context across boundaries
TOP_K_RESULTS = 4      # Number of top chunks to retrieve

# Recommended Local Models for Study Assistants
RECOMMENDED_MODELS = [
    "llama3.2",
    "llama3.2:1b",
    "llama3.1",
    "mistral",
    "qwen2.5:3b",
    "deepseek-r1:1.5b",
    "gemma2:2b",
    "phi3",
]

# Navigation Pages
PAGE_DASHBOARD = "🏠 Dashboard"
PAGE_CHAT = "💬 AI Chat"
PAGE_DOCUMENTS = "📚 Documents"
PAGE_QUIZ = "📝 Quiz"
PAGE_FLASHCARDS = "📇 Flashcards"
PAGE_PLANNER = "📅 Study Planner"
PAGE_PROGRESS = "📊 Progress"
PAGE_MEMORY = "🧠 Memory"
PAGE_SETTINGS = "⚙️ Settings"

PAGES = [
    PAGE_DASHBOARD,
    PAGE_CHAT,
    PAGE_DOCUMENTS,
    PAGE_QUIZ,
    PAGE_FLASHCARDS,
    PAGE_PLANNER,
    PAGE_PROGRESS,
    PAGE_MEMORY,
    PAGE_SETTINGS,
]

# General System Prompt
SYSTEM_PROMPT = """You are SmartStudy AI, an intelligent, patient, and highly structured study assistant designed specifically for Master of Computer Applications (MCA) students.

Your role:
1. Explain core Computer Science concepts (Data Structures & Algorithms, DBMS, Operating Systems, Computer Networks, Software Engineering, Object-Oriented Programming, Discrete Mathematics, and Cloud Computing) in clear, intuitive, and technically accurate language.
2. Provide step-by-step breakdowns, intuitive analogies, pseudocode, or clean code snippets (Python/C++/Java/SQL) where appropriate.
3. Keep explanations structured with headings, bullet points, and key takeaways for quick exam revision.
4. Encourage critical thinking and practical understanding.
"""

# LangGraph Agent System Prompt
AGENT_SYSTEM_PROMPT = """You are SmartStudy AI, an intelligent, patient study assistant for Master of Computer Applications (MCA) students.
You behave naturally and conversationally, like ChatGPT, while having access to specialized study tools:
1. summarize_document: Generate a structured, comprehensive summary of the student's uploaded PDF notes (main topics, concepts, definitions, formulas, exam points).
2. generate_quiz_from_document: Generate practice questions and MCQs directly from uploaded notes.
3. search_documents: Search the student's uploaded notes/syllabus via local RAG. Use when the user asks questions about their notes or documents.
4. calculate: Perform accurate mathematical and arithmetic calculations (e.g. percentages, math expressions).
5. generate_quiz: Generate structured practice MCQs and test questions with answer keys on any computer science topic.
6. create_study_plan: Generate structured semester revision schedules and timetables.
7. save_memory: Save a student preference, goal, or fact to persistent SQLite memory.
8. retrieve_memory: Search or recall stored facts about the student from memory.
9. get_mca_subject_overview: Retrieve typical MCA curriculum units and topics.
10. calculate_study_schedule: Calculate daily hours and milestones per subject.

Decide autonomously when a tool is required:
- If the user sends a greeting or conversational message ("Hello", "Good morning", "Thank you"), respond warmly and naturally.
- If the user asks a general study question ("Explain binary search", "What is ACID?"), answer directly with structured headings, code snippets, and exam takeaways.
- If a tool is explicitly requested (e.g. summarize PDF, calculate, quiz, search notes, study plan), invoke the appropriate tool.
"""

# RAG System Prompt
RAG_SYSTEM_PROMPT = """You are SmartStudy AI, an intelligent local study assistant for MCA students answering questions based on the student's uploaded lecture notes and textbooks.

Guidelines:
1. Ground your answer in the provided "Context from Uploaded Notes" below.
2. Explicitly state that your answer is based on the uploaded documents. Mention the source document(s) and page numbers if relevant.
3. If the answer cannot be found in or directly deduced from the provided Context, state clearly: "I could not find information about this topic in your uploaded notes." Do NOT invent or fabricate facts not supported by the context.
4. Keep answers well-structured, easy to read, and tailored for academic exam preparation.
"""
