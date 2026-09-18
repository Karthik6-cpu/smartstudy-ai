# 🎓 SmartStudy AI — Local AI Study Assistant for MCA Students

**SmartStudy AI** is a production-grade, 100% local, privacy-first AI study assistant engineered specifically for **Master of Computer Applications (MCA)** students. It combines an autonomous **LangGraph** state machine workflow, **LangChain** tool orchestration, a local **PDF Retrieval-Augmented Generation (RAG)** engine with persistent **FAISS**, a safe AST mathematical calculator, interactive quizzes, an actionable semester study planner, persistent SQLite memory, and visual learning analytics — all powered entirely by local **Ollama** LLMs without requiring an OpenAI API key or cloud subscription.

---

## 📑 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Features Matrix](#2-features-matrix)
3. [Architecture Diagram](#3-architecture-diagram)
4. [Installation & Setup](#4-installation--setup)
5. [Ollama Setup Guide](#5-ollama-setup-guide)
6. [Model Setup & Hardware Recommendations](#6-model-setup--hardware-recommendations)
7. [Local RAG Pipeline Deep-Dive](#7-local-rag-pipeline-deep-dive)
8. [LangChain Orchestration Deep-Dive](#8-langchain-orchestration-deep-dive)
9. [LangGraph State Machine & Cyclic Routing](#9-langgraph-state-machine--cyclic-routing)
10. [Tool Calling & Safe AST Calculator](#10-tool-calling--safe-ast-calculator)
11. [Database & Persistence Architecture](#11-database--persistence-architecture)
12. [Project Structure](#12-project-structure)
13. [Running Instructions](#13-running-instructions)
14. [Troubleshooting & Diagnostics](#14-troubleshooting--diagnostics)
15. [Future Improvements Roadmap](#15-future-improvements-roadmap)

---

## 1. Project Overview

Master of Computer Applications (MCA) curriculums demand rigorous mastery over theoretical concepts, mathematical problem-solving, and practical coding across core domains:
- Data Structures & Algorithms (Trees, Graphs, Dynamic Programming)
- Database Management Systems (Normalization, SQL Joins, Concurrency)
- Operating Systems (Deadlocks, Process Synchronization, Paging)
- Computer Networks (TCP/IP, Routing Protocols, OSI Model)
- Software Engineering (SDLC, Agile Scrum, CMMI Maturity Models)

Traditional cloud-based AI tools pose privacy concerns, cost recurring subscription fees, hallucinate outside textbook contexts, and fail when offline. **SmartStudy AI** solves this by running 100% locally on student hardware, grounding answers in their professor's uploaded lecture slides and course notes, and providing trackable assessments and study timetables.

---

## 2. Features Matrix

| Feature | Description | Engine / Tech |
| :--- | :--- | :--- |
| **🔒 100% Local & Private** | Zero data sent to cloud servers; no API keys required. | Local Ollama |
| **🤖 Autonomous Agent** | Evaluates user prompts to decide whether to answer directly or trigger specialized tools. | LangGraph + LangChain |
| **📚 Document RAG** | Upload PDFs, extract text, generate 384-dim dense embeddings, and search via FAISS. | PyPDF + SentenceTransformers + FAISS |
| **🧮 Safe Math Calculator** | AST-based arithmetic and percentage evaluation with zero arbitrary execution. | Python AST Parser |
| **📝 Interactive Quizzes** | Generate MCQs from topics or grounded in uploaded PDFs; instant grading and review. | LangChain + SQLite |
| **📅 Study Planner** | Generate semester timetables with workload estimation and **⭕ / ⏳ / ✅ status tracking**. | LangGraph Planner Tool + SQLite |
| **🧠 Student Memory** | Stores student learning profile, preferred languages, and target exam goals across sessions. | SQLite (`student_memory.db`) |
| **📊 Learning Analytics** | Track syllabus completion %, quiz score trends, subject coverage, and activity logs. | Pandas + Streamlit Charts |
| **🔍 Developer / Debug Mode** | Live visual audit trail of the agent's internal routing, tool inputs, and results. | Streamlit Toggle |

---

## 3. Architecture Diagram

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        Streamlit Frontend Layer                        │
│ (app.py: Dashboard, AI Chat, Documents, Quiz, Planner, Memory, Progress)│
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ User Request
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   LangGraph StateGraph Workflow                        │
│                      (graph/study_graph.py)                            │
│                                                                        │
│   START ──► [agent_node] ──► should_continue?                          │
│                                ├── 'tools' ──► [safe_tool_executor]    │
│                                │                     │                 │
│                                │                     ▼                 │
│                                │                 [agent_node]          │
│                                └── 'final' ──► [final_response] ──► END│
└──────────────┬───────────────────────────────────────────┬─────────────┘
               │                                           │
   [Tool Calls Triggered]                       [Conversation State]
               ▼                                           ▼
┌───────────────────────────────┐           ┌────────────────────────────┐
│      LangChain Tools          │           │   SQLite Storage Layer     │
│ ├── search_documents (RAG)    │           │  (data/memory/)            │
│ ├── calculate (Safe AST)      │           │  ├── graph_checkpoints.db  │
│ ├── generate_quiz (MCQs)      │           │  ├── student_memory.db     │
│ ├── create_study_plan         │           │  └── study_data.db         │
│ └── save / retrieve_memory    │           └────────────────────────────┘
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐           ┌────────────────────────────┐
│      Local RAG Pipeline       │           │    Local Ollama Engine     │
│ ├── PyPDF Text Extraction     │           │  (http://localhost:11434)  │
│ ├── all-MiniLM-L6-v2 Embeddings│          │  • llama3.2 (Default)      │
│ └── FAISS Flat-IP Vector Store│           │  • Streaming token output  │
└───────────────────────────────┘           └────────────────────────────┘
```

---

## 4. Installation & Setup

### Prerequisites
- Operating System: Windows 10/11, macOS, or Linux
- Python: Version **3.11** or **3.12** installed
- Git (optional)

### Step 1: Clone or Navigate to Project
```bash
cd smartstudy-ai
```

### Step 2: Create & Activate a Virtual Environment
```powershell
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 5. Ollama Setup Guide

Ollama provides the local hardware inference runtime for open-weights LLMs.

1. **Download:** Visit [ollama.com/download](https://ollama.com/download) and run the installer for your OS.
2. **Verify Installation:**
   ```bash
   ollama --version
   ```
3. **Start the Ollama Daemon:**
   - **Windows / macOS:** Launch the desktop Ollama application from the Start Menu / Applications.
   - **Linux / Headless Terminal:** Run `ollama serve` in a background terminal.

---

## 6. Model Setup & Hardware Recommendations

SmartStudy AI uses **`llama3.2`** by default (Meta's 3-Billion parameter lightweight instruction-tuned model):

```bash
ollama pull llama3.2
```

### Hardware Recommendations:

| Hardware | RAM | Recommended Model | Ollama Pull Command |
| :--- | :--- | :--- | :--- |
| Standard Laptop | 8 GB | `llama3.2:1b` or `qwen2.5:1.5b` | `ollama pull llama3.2:1b` |
| Recommended | 16 GB | `llama3.2` (Default) | `ollama pull llama3.2` |
| High Performance | 32 GB+ / Dedicated GPU | `llama3.1` (8B) or `deepseek-r1:8b` | `ollama pull llama3.1` |

You can switch models at any time under **⚙️ Settings** in the application.

---

## 7. Local RAG Pipeline Deep-Dive

The Retrieval-Augmented Generation (RAG) system runs completely offline:

1. **Document Ingestion (`rag/pdf_processor.py`):**
   - Ingests multiple PDF files via Streamlit file uploader.
   - Extracts text page-by-page using `pypdf.PdfReader`.
   - Cleans null bytes, non-printable characters, and excessive whitespace.
   - Splits text into overlapping chunks (default: 600 characters with 100 character overlap) while respecting paragraph and sentence boundaries.
   - Attaches rich metadata: `{"source": filename, "page": page_number, "chunk_index": idx}`.
2. **Neural Embeddings (`rag/embeddings.py`):**
   - Uses `sentence-transformers` with `all-MiniLM-L6-v2`.
   - Encodes text chunks into 384-dimensional dense vectors normalized for cosine similarity via inner product.
   - Cached singleton instance prevents reloading model weights across user queries.
3. **Persistent Vector Storage (`rag/vector_store.py`):**
   - High-performance `faiss.IndexFlatIP` index.
   - Persisted to disk at `data/vector_store/index.faiss` and `metadata.json`.
   - Rebuilt automatically when documents are deleted.
4. **Retrieval & Citations (`rag/retriever.py`):**
   - Custom `LangChainFAISSRetriever` inheriting from `langchain_core.retrievers.BaseRetriever`.
   - Provides citations with source filename, page number, and text excerpt in expandable UI accordions.

---

## 8. LangChain Orchestration Deep-Dive

SmartStudy AI uses modern LangChain (`langchain-core>=0.2.0`, `langchain-ollama>=0.1.0`):
- **Reusable Prompt Templates (`prompts/templates.py`):**
  - `GENERAL_STUDY_PROMPT`: MCA pedagogical persona with code snippets and revision takeaways.
  - `RAG_STUDY_PROMPT`: Context-grounded instructions with anti-hallucination guardrails.
  - `QUIZ_GENERATION_PROMPT`: MCQ formatting rules with answer keys and explanations.
  - `STUDY_PLANNER_PROMPT`: Spaced repetition revision timetable formulas.
- **LCEL Chains (`chains/study_chains.py`):**
  - Composable pipelines using pipe syntax: `prompt | llm | StrOutputParser()`.
  - Native streaming token support passed directly to Streamlit's `st.write_stream`.

---

## 9. LangGraph State Machine & Cyclic Routing

The central agent is structured as an explicit **LangGraph StateGraph** (`graph/study_graph.py`):

### 1. Typed State (`StudyAgentState`):
```python
class StudyAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_message: str
    selected_tools: List[str]
    retrieved_documents: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    final_response: str
    debug_trace: List[Dict[str, Any]]
    error: Optional[str]
```

### 2. Graph Topology:
- **`agent_node`:** Calls `ChatOllama.bind_tools(ALL_STUDY_TOOLS)` with heuristic intent routing fallback.
- **`should_continue`:** Conditional router that inspects `last_message.tool_calls`.
  - If tools called: routes to `tools`.
  - If no tools called: routes to `final_response`.
- **`safe_tool_executor` (ToolNode):** Executes tools, captures any exception politely, and feeds `ToolMessage` back to the agent for synthesis without exposing raw tracebacks.
- **`final_response_node`:** Extracts final answer and citation metadata.

---

## 10. Tool Calling & Safe AST Calculator

SmartStudy AI provides 5 specialized tools registered with `@tool`:

1. **`search_documents(query)`:** Queries FAISS vector store for relevant lecture notes.
2. **`calculate(expression)`:** Safely evaluates arithmetic expressions using Python's Abstract Syntax Tree (`ast`).
   - Supports percentages (e.g. `"25% of 480"` $\rightarrow$ `120.0`), powers (`^`), square roots (`sqrt`), and trigonometry.
   - **Safety Guarantee:** Strict node whitelist. **Zero `eval()` or `exec()`**. Rejects imports, variables, file access, or system calls.
3. **`generate_quiz(topic, number_of_questions, difficulty)`:** Generates structured MCQs and answer keys.
4. **`create_study_plan(subjects, available_hours, exam_date)`:** Builds structured revision schedules.
5. **`save_memory(key, value)` & `retrieve_memory(query)`:** CRUD operations against SQLite.

---

## 11. Database & Persistence Architecture

Data is stored persistently in `smartstudy-ai/data/`:

```text
data/
├── documents/                  # Local uploaded PDF copies
├── vector_store/
│   ├── index.faiss             # FAISS Flat-IP dense vector index
│   ├── metadata.json           # Chunk text, filenames, pages
│   └── embeddings.npy          # Raw vector cache for fast rebuilds
└── memory/
    ├── student_memory.db       # Key-value student preferences & profile
    ├── study_data.db           # Quiz history, study tasks & activity logs
    └── graph_checkpoints.db    # LangGraph multi-turn session checkpoints
```

### SQLite Schemas:
- **`student_memory`**: `key TEXT PRIMARY KEY, value TEXT, updated_at TIMESTAMP`
- **`quiz_history`**: `id, topic, difficulty, document_name, total_questions, score, percentage, details_json, created_at`
- **`study_tasks`**: `id, plan_id, subject, topic, priority, status ('Not Started', 'In Progress', 'Completed'), exam_date, allocated_hours, updated_at`
- **`activity_log`**: `id, activity_type, description, created_at`

---

## 12. Project Structure

```text
smartstudy-ai/
├── app.py                         # Main Streamlit application & routing
├── requirements.txt               # Complete production dependencies
├── README.md                      # Production documentation
├── config/
│   └── settings.py                # Configuration paths, models, and prompts
├── memory/
│   ├── __init__.py
│   ├── sqlite_memory.py           # Student profile memory store
│   └── storage_manager.py         # Unified SQLite manager (quizzes, tasks, logs)
├── chains/
│   ├── __init__.py
│   ├── agent.py                   # Intent routing fallback & tool executor
│   ├── quiz_generator.py          # Structured JSON quiz generator with fallbacks
│   └── study_chains.py            # LCEL prompt pipelines (Quiz, Planner)
├── graph/
│   ├── __init__.py
│   ├── state.py                   # Typed StudyAgentState definition
│   ├── nodes.py                   # agent_node, safe_tool_executor, final_response_node
│   └── study_graph.py             # StateGraph assembly, compilation & runner
├── tools/
│   ├── __init__.py                # Tools registry
│   ├── safe_calculator.py         # AST mathematical parser (zero eval)
│   └── study_tools.py             # LangChain @tool definitions (5 tools)
├── prompts/
│   ├── __init__.py
│   └── templates.py               # Reusable LangChain ChatPromptTemplates
├── rag/
│   ├── __init__.py
│   ├── pdf_processor.py           # PyPDF text extraction & chunking
│   ├── embeddings.py              # SentenceTransformer embedding wrapper
│   ├── vector_store.py            # Persistent FAISS index and metadata
│   └── retriever.py               # LangChainFAISSRetriever (BaseRetriever)
├── llm/
│   ├── ollama_client.py           # Native Ollama client wrapper with diagnostics
│   └── langchain_client.py        # Modern ChatOllama factory
├── ui/
│   ├── dashboard.py               # System health & architecture overview
│   ├── chat.py                    # LangGraph Chat UI with Debug Mode toggle
│   ├── documents.py               # PDF upload, management & deletion
│   ├── quiz.py                    # Interactive Quiz page with scoring & history
│   ├── planner.py                 # Study Planner with interactive task tracker
│   ├── memory.py                  # Student Memory & Profile manager
│   ├── progress.py                # Analytics dashboard with charts & activity feed
│   └── settings.py                # Model selection and connection settings
├── scripts/
│   └── generate_sample_notes.py   # Script to generate sample MCA CMMI notes
└── tests/
    ├── __init__.py
    ├── run_all_tests.py           # Master production scorecard runner
    ├── test_full_features.py      # Quiz scoring, task status, and analytics tests
    ├── test_langgraph_workflow.py # LangGraph StateGraph, nodes, and routing tests
    ├── test_tools_and_agent.py    # Safe calculator, SQLite memory & tool tests
    ├── test_langchain_integration.py # Prompt and LCEL chain tests
    ├── test_rag_pipeline.py       # Full PDF -> FAISS -> Retrieval test suite
    └── test_ollama.py             # Native Ollama client tests
```

---

## 13. Running Instructions

### 1. Launch the Streamlit Web Application
From the `smartstudy-ai` root directory:

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

### 2. Run Automated Verification Tests
To run the full production test suite:

```bash
python tests/run_all_tests.py
```

---

## 14. Troubleshooting & Diagnostics

| Symptom | Probable Cause | Actionable Solution |
| :--- | :--- | :--- |
| **"Ollama Service Unavailable"** | Ollama daemon is not running | Launch the desktop Ollama app or run `ollama serve` in a terminal window. |
| **"Model Not Found Locally"** | Target model has not been downloaded | Run `ollama pull llama3.2` (or select another installed model under **⚙️ Settings**). |
| **"No extractable text found in PDF"** | Uploaded file contains scanned images without OCR | Ensure PDF contains digital selectable text, or run OCR prior to uploading. |
| **Out of Memory / High RAM Usage** | Model parameters too large for hardware | Run `ollama pull llama3.2:1b` and select it under **⚙️ Settings**. |
| **Slow Embedding Generation** | CPU running many threads on large PDF | `all-MiniLM-L6-v2` is optimized for CPU; chunking in 600-char intervals avoids overhead. |

---

## 15. Future Improvements Roadmap

- [ ] **OCR Engine Integration:** Add local `pytesseract` or `surya-ocr` for scanned paper and handwritten lecture notes.
- [ ] **Flashcards & Anki Export:** Export generated quiz MCQs directly into standard `.apkg` flashcard decks.
- [ ] **Voice Interaction:** Integrate local Whisper (`faster-whisper`) for speech-to-text question entry.
- [ ] **Multi-Agent Debate:** Add specialized LangGraph agent subgraphs (e.g. Code Reviewer Agent, Exam Grader Agent).
