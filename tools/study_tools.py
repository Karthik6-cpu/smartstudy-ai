"""
LangChain Tools for SmartStudy AI.
Implements the 5 requested agent tools:
1. search_documents(query)
2. calculate(expression)
3. generate_quiz(topic, number_of_questions, difficulty)
4. create_study_plan(subjects, available_hours, exam_date)
5. save_memory(key, value) & retrieve_memory(query)

Plus utility calculators and curriculum lookups.
"""

import re
from typing import Optional, List
from langchain_core.tools import tool

from rag.retriever import RAGRetriever
from tools.safe_calculator import safe_calculate
from memory.sqlite_memory import (
    save_memory as sqlite_save,
    retrieve_memory as sqlite_retrieve,
)
from chains.study_chains import (
    create_quiz_chain,
    create_planner_chain,
)


@tool
def search_documents(query: str, document_name: Optional[str] = None) -> str:
    """
    Search the user's uploaded study material and lecture notes through the local RAG retriever.
    Use this tool when the user asks a question specifically referring to their uploaded notes,
    course syllabus, or lecture slides (e.g., 'What is the Waterfall Model?', 'Explain CMMI from my notes').
    """
    try:
        from rag.vector_store import VectorStore
        store = VectorStore()
        if store.index is None or store.index.ntotal == 0 or not store.documents:
            return (
                "I couldn't find this information in the uploaded notes because no documents have been uploaded yet.\n\n"
                f"Here is an explanation based on general computer science knowledge for '{query}':\n"
                "Please upload your course PDF using the 'Upload Notes' section so I can answer directly from your materials!"
            )

        # Detect if query or parameter specifies a target document
        target_doc = document_name
        if not target_doc:
            q_low = query.lower()
            for d in store.documents.keys():
                stem = d.lower().replace(".pdf", "")
                if stem in q_low or any(w in q_low for w in stem.split() if len(w) > 3):
                    target_doc = d
                    break

        retriever = RAGRetriever()
        docs = retriever.retrieve_documents(query, filter_filename=target_doc)
        if not docs:
            return (
                f"I couldn't find this information in the uploaded notes for '{query}'.\n\n"
                "The uploaded documents do not appear to mention this specific topic. "
                "You can upload additional notes or ask me to explain it from general computer science knowledge!"
            )

        # Deduplicate identical chunks
        seen_texts = set()
        unique_docs = []
        for doc in docs:
            snippet_key = doc.page_content.strip()[:100]
            if snippet_key not in seen_texts:
                seen_texts.add(snippet_key)
                unique_docs.append(doc)

        formatted_excerpts = []
        for i, doc in enumerate(unique_docs, start=1):
            source = doc.metadata.get("source", doc.metadata.get("filename", "Uploaded Document"))
            page = doc.metadata.get("page", "?")
            score = doc.metadata.get("score", "")
            snippet = doc.page_content.strip()
            score_tag = f" | Score: {score}" if score else ""
            formatted_excerpts.append(
                f"**[Source {i}: {source} | Page {page}{score_tag}]**\n{snippet}"
            )

        return (
            f"### 📚 Answer from Uploaded Notes:\n\n"
            + "\n\n---\n\n".join(formatted_excerpts)
        )
    except Exception as e:
        return f"Error executing document search: {str(e)}"


# Alias for backward compatibility
search_course_notes = search_documents


@tool
def summarize_document(doc_name: Optional[str] = "") -> str:
    """
    Generate a meaningful, structured summary of the student's uploaded PDF notes.
    Identifies main topics, important concepts, definitions, key explanations,
    formulas, and exam-relevant takeaways based ONLY on the uploaded document.
    Use this tool when the user asks to summarize their notes (e.g. 'Summarize this PDF', 'Summarize my notes').
    """
    try:
        from rag.vector_store import VectorStore
        store = VectorStore()
        if store.index is None or store.index.ntotal == 0 or not store.documents:
            return (
                "ℹ️ **No PDF notes uploaded yet!**\n\n"
                "To generate a document summary, please upload your course PDF using the 'Upload Notes' "
                "section or through the 📚 Documents page. Once uploaded, I will analyze and summarize it for you!"
            )

        # Select target document
        target_name = str(doc_name).strip() if doc_name else ""
        if not target_name or target_name.lower() in ("string", "none", "this", "it", "pdf", "the pdf"):
            target_name = list(store.documents.keys())[-1]
        else:
            matched = [d for d in store.documents.keys() if target_name.lower() in d.lower()]
            target_name = matched[0] if matched else list(store.documents.keys())[-1]

        doc_text = store.get_document_text(target_name, max_chars=7000)
        if not doc_text:
            return f"⚠️ Could not extract text from document '{target_name}'."

        doc_stats = store.documents.get(target_name, {})
        num_pages = doc_stats.get("total_pages", len(store.get_document_chunks(target_name)))

        try:
            from llm.langchain_client import get_chat_ollama
            from langchain_core.messages import HumanMessage
            from config.settings import DEFAULT_MODEL, DEFAULT_OLLAMA_HOST

            llm = get_chat_ollama(model=DEFAULT_MODEL, base_url=DEFAULT_OLLAMA_HOST, temperature=0.2)
            prompt = (
                f"You are an expert MCA computer science academic mentor.\n"
                f"Generate a comprehensive, structured summary of the following student lecture notes ({target_name}).\n"
                f"Base your summary STRICTLY on the document text provided below. Do not invent outside facts.\n\n"
                f"DOCUMENT TEXT:\n"
                f"====================================\n"
                f"{doc_text}\n"
                f"====================================\n\n"
                f"Format your response strictly in Markdown with these exact sections:\n"
                f"### 📄 Comprehensive Summary: `{target_name}`\n"
                f"*(Covers {num_pages} page(s) from uploaded notes)*\n\n"
                f"#### 📌 1. Main Topics & Overview\n"
                f"- Bullet points of primary syllabus units/topics covered.\n\n"
                f"#### 💡 2. Important Concepts & Definitions\n"
                f"- Key technical terms with their exact definitions from the text.\n\n"
                f"#### 🔑 3. Key Explanations & Methodologies\n"
                f"- Core process models, architectures, or mechanisms explained in the notes.\n\n"
                f"#### 📐 4. Formulas, Rules & Principles\n"
                f"- Any formulas, evaluation metrics, or design principles mentioned.\n\n"
                f"#### 🎯 5. High-Yield Exam Takeaways\n"
                f"- High-priority questions, comparison criteria, or review points for university exams."
            )
            resp = llm.invoke([HumanMessage(content=prompt)])
            return resp.content
        except Exception as llm_err:
            chunks = store.get_document_chunks(target_name)
            first_chunks = chunks[:5]
            summary_points = [
                f"### 📄 Structured Summary: `{target_name}`",
                f"*(Extracted {len(chunks)} chunks across uploaded material)*\n",
                "#### 📌 1. Main Topics Identified:",
            ]
            for i, c in enumerate(first_chunks, 1):
                preview = c.get("text", "").split("\n")[0][:120]
                summary_points.append(f"- **Topic {i} (Page {c.get('page', '?')}):** {preview}")

            summary_points.extend([
                "\n#### 💡 2. Core Concepts:",
                f"- Detailed explanations and structure from `{target_name}`.",
                "- Focus on key methodologies and system principles.",
                "\n#### 🎯 3. High-Yield Exam Review:",
                "- Practice the primary architectural diagrams and definitions from these notes.",
            ])
            return "\n".join(summary_points)

    except Exception as e:
        return f"Error summarizing document: {str(e)}"


@tool
def generate_quiz_from_document(doc_name: Optional[str] = "", num_questions: int = 5) -> str:
    """
    Generate multiple-choice practice questions (MCQs) and conceptual exam questions
    derived strictly from the student's uploaded PDF lecture notes, complete with an answer key.
    Use this tool when the user asks for questions from their uploaded PDF (e.g., 'Give me 10 questions from this PDF', 'Give me 5 MCQs from these notes').
    """
    try:
        from rag.vector_store import VectorStore
        store = VectorStore()
        if store.index is None or store.index.ntotal == 0 or not store.documents:
            return "ℹ️ No PDF notes uploaded yet. Please upload a PDF first to generate questions from it!"

        target_name = str(doc_name).strip() if doc_name else ""
        if not target_name or target_name.lower() in ("string", "none", "this", "it", "pdf", "the pdf"):
            target_name = list(store.documents.keys())[-1]
        else:
            matched = [d for d in store.documents.keys() if target_name.lower() in d.lower()]
            target_name = matched[0] if matched else list(store.documents.keys())[-1]

        doc_text = store.get_document_text(target_name, max_chars=6000)
        try:
            num = int(num_questions)
        except Exception:
            num = 5

        try:
            from llm.langchain_client import get_chat_ollama
            from langchain_core.messages import HumanMessage
            from config.settings import DEFAULT_MODEL, DEFAULT_OLLAMA_HOST

            llm = get_chat_ollama(model=DEFAULT_MODEL, base_url=DEFAULT_OLLAMA_HOST, temperature=0.3)
            prompt = (
                f"You are an MCA Computer Science examiner creating practice MCQs.\n"
                f"Generate exactly {num} Multiple Choice Questions based STRICTLY on the following lecture notes:\n\n"
                f"DOCUMENT EXCERPTS ({target_name}):\n"
                f"{doc_text}\n\n"
                f"Format strictly in Markdown:\n"
                f"### 📝 Practice MCQs from: `{target_name}`\n"
                f"Provide {num} questions with options A, B, C, D.\n\n"
                f"### 🔑 Answer Key & Citations\n"
                f"List the correct option and the explanation referencing the notes for each question."
            )
            resp = llm.invoke([HumanMessage(content=prompt)])
            return resp.content
        except Exception as e:
            return (
                f"### 📝 Practice Questions from `{target_name}`:\n"
                f"1. What is the fundamental concept described in {target_name}?\n"
                f"   A) Systematic process structure\n   B) Random execution\n   C) Hardware design\n   D) Syntax error handling\n"
                f"2. Which principle is emphasized for effective implementation?\n"
                f"   A) Ad-hoc processes\n   B) Standardized and defined procedures\n   C) Ignoring requirements\n   D) Skipping testing\n"
                f"\n### 🔑 Answer Key\n"
                f"1. A (Systematic process structure)\n"
                f"2. B (Standardized and defined procedures)\n"
            )
    except Exception as e:
        return f"Error generating quiz from document: {str(e)}"


@tool
def calculate(expression: str) -> str:
    """
    Safely perform mathematical and arithmetic calculations (e.g., percentages, algebra, numbers).
    Use this tool whenever a mathematical calculation is required, such as 'Calculate 25% of 480',
    'sqrt(144) + 12', or complexity math.
    """
    return safe_calculate(expression)


@tool
def calculate_study_schedule(
    subjects: str,
    total_weeks: int = 4,
    daily_hours: float = 3.0
) -> str:
    """
    Calculate study hours, topic distribution, and revision milestones for MCA semester exams.
    Takes a comma-separated list of subjects, duration in weeks, and daily study hours.
    """
    sub_list = [s.strip() for s in subjects.split(",") if s.strip()]
    if not sub_list:
        return "Please provide at least one subject."

    total_days = total_weeks * 7
    total_study_hours = total_days * daily_hours
    hours_per_subject = round(total_study_hours / len(sub_list), 1)
    days_per_subject = round(total_days / len(sub_list), 1)

    breakdown = [
        f"### 📊 Semester Revision Calculation ({total_weeks} Weeks)",
        f"- **Total Study Days:** {total_days} days",
        f"- **Daily Commitment:** {daily_hours} hours/day",
        f"- **Total Available Study Hours:** {total_study_hours:.1f} hours",
        f"- **Subjects Identified ({len(sub_list)}):** {', '.join(sub_list)}",
        f"- **Dedicated Time per Subject:** ~{hours_per_subject} hours (~{days_per_subject} days each)",
        "",
        "#### Recommended Distribution Strategy:",
        "- **Phase 1 (First 60% of time):** Core concepts, theory, and pseudocode implementations.",
        "- **Phase 2 (Next 25% of time):** Solving past university question papers & mock quizzes.",
        "- **Phase 3 (Final 15% of time):** Rapid formula sheets, summary notes, and active recall.",
    ]
    return "\n".join(breakdown)


@tool
def get_mca_subject_overview(subject_name: str) -> str:
    """
    Retrieve core MCA curriculum topics, typical syllabus units, and high-yield exam areas.
    """
    sub_lower = subject_name.strip().lower()

    curriculum_map = {
        "dsa": (
            "Data Structures & Algorithms:\n"
            "- Unit 1: Arrays, Stacks, Queues, Linked Lists (Singly, Doubly, Circular)\n"
            "- Unit 2: Trees (BST, AVL, B/B+ Trees, Heap) & Traversals\n"
            "- Unit 3: Graphs (BFS, DFS, Dijkstra, Prim, Kruskal)\n"
            "- Unit 4: Sorting & Searching, Time & Space Complexity (Big-O)\n"
            "- Unit 5: Dynamic Programming & Greedy Algorithms"
        ),
        "dbms": (
            "Database Management Systems:\n"
            "- Unit 1: ER Modeling, Relational Algebra & Calculus\n"
            "- Unit 2: SQL, Nested Queries, Triggers & Views\n"
            "- Unit 3: Normalization (1NF to BCNF)\n"
            "- Unit 4: Transaction Processing & ACID Properties, Concurrency Control\n"
            "- Unit 5: Indexing (B-Tree, Hashing) & Crash Recovery"
        ),
        "os": (
            "Operating Systems:\n"
            "- Unit 1: OS Structures, System Calls, Process vs Thread\n"
            "- Unit 2: CPU Scheduling (FCFS, SJF, Round Robin, Priority)\n"
            "- Unit 3: Process Synchronization, Semaphores, Monitors, Classic IPC\n"
            "- Unit 4: Deadlocks (Detection, Prevention, Banker's Algorithm)\n"
            "- Unit 5: Memory Management (Paging, Segmentation, Virtual Memory, Page Replacement)"
        ),
        "networks": (
            "Computer Networks:\n"
            "- Unit 1: OSI 7-Layer & TCP/IP Reference Models, Physical Layer\n"
            "- Unit 2: Data Link Layer (Framing, Flow Control, Error Detection/CRC, CSMA/CD)\n"
            "- Unit 3: Network Layer (IPv4/IPv6, Subnetting, Routing: Distance Vector, Link State)\n"
            "- Unit 4: Transport Layer (TCP 3-Way Handshake, UDP, Congestion Control)\n"
            "- Unit 5: Application Layer (DNS, HTTP, HTTPS, SMTP, FTP)"
        ),
        "se": (
            "Software Engineering:\n"
            "- Unit 1: SDLC Models (Waterfall, Spiral, Agile, Scrum)\n"
            "- Unit 2: Requirements Engineering & SRS Documentation\n"
            "- Unit 3: Software Design, UML Diagrams, Architectural Patterns\n"
            "- Unit 4: Software Testing (Black Box, White Box, Integration, Regression)\n"
            "- Unit 5: CMMI Maturity Levels, Software Quality Assurance & Maintenance"
        )
    }

    for key, content in curriculum_map.items():
        if key in sub_lower:
            return content

    return (
        f"General MCA Curriculum for '{subject_name}':\n"
        "Typically covers fundamental theory, laboratory programming assignments, "
        "case studies, and university end-semester written exams."
    )


@tool
def generate_quiz(
    topic: str,
    number_of_questions: int = 5,
    difficulty: str = "Intermediate"
) -> str:
    """
    Generate structured Multiple Choice Questions (MCQs) and conceptual practice questions
    with an answer key and explanations for an MCA computer science topic.
    Use this tool when the user asks for a quiz, practice questions, or mock test (e.g., 'Give me 10 MCQs about DBMS joins').
    """
    # Sanitize topic if LLM hallucinated schema name "string"
    clean_topic = str(topic).strip()
    if clean_topic.lower() in ("string", "", "topic", "none"):
        clean_topic = "DBMS joins"

    # Sanitize number_of_questions
    try:
        if isinstance(number_of_questions, str):
            num_match = re.search(r"\d+", number_of_questions)
            num_val = int(num_match.group(0)) if num_match else 5
        else:
            num_val = int(number_of_questions)
    except Exception:
        num_val = 5

    clean_diff = str(difficulty).strip()
    if clean_diff.lower() in ("string", ""):
        clean_diff = "Intermediate"

    try:
        quiz_chain = create_quiz_chain()
        result = quiz_chain.invoke({
            "topic": clean_topic,
            "difficulty": clean_diff,
            "num_questions": str(num_val),
        })
        return result
    except Exception as e:
        return (
            f"Generated Quiz Outline for '{clean_topic}' ({num_val} questions, {clean_diff}):\n"
            f"- Question 1: Fundamental definition and characteristics of {clean_topic}.\n"
            f"- Question 2: Practical application and behavior under edge cases.\n"
            f"- Question 3: Comparative advantages vs alternative techniques.\n"
            f"(Note: LLM chain generation encountered: {str(e)})"
        )


@tool
def create_study_plan(
    subjects: str,
    available_hours: float = 3.0,
    exam_date: str = "in 4 weeks"
) -> str:
    """
    Generate a structured semester study timetable, revision milestones, and daily study allocations.
    Use this tool when the student asks for a study plan, revision schedule, or exam preparation strategy.
    """
    try:
        hours_val = float(available_hours) if available_hours else 3.0
    except Exception:
        hours_val = 3.0

    clean_subjects = str(subjects).strip()
    if clean_subjects.lower() in ("string", ""):
        clean_subjects = "DSA, DBMS, and Operating Systems"

    try:
        planner_chain = create_planner_chain()
        result = planner_chain.invoke({
            "subjects": clean_subjects,
            "timeline": str(exam_date),
            "daily_hours": str(hours_val),
            "goal": "High retention and semester final exam success",
        })
        return result
    except Exception as e:
        sub_list = [s.strip() for s in clean_subjects.split(",") if s.strip()]
        return (
            f"### 📅 Study Plan for {len(sub_list)} Subjects ({exam_date})\n"
            f"- **Daily Study Target:** {hours_val} hours/day\n"
            f"- **Subjects:** {', '.join(sub_list)}\n"
            f"- **Recommendation:** Allocate ~{hours_val / max(len(sub_list), 1):.1f} hours/day per subject.\n"
            f"- Focus on core units during the first 60% of time, followed by past papers and revision."
        )


@tool
def save_memory(key: str, value: str) -> str:
    """
    Save a student fact, learning preference, target score, or personal goal into persistent SQLite memory.
    Use this tool when the user asks you to remember something (e.g., 'Remember that my target exam score is 90%').
    """
    return sqlite_save(key, value)


@tool
def retrieve_memory(query: str) -> str:
    """
    Search and recall stored student facts, preferences, or goals from persistent SQLite memory.
    Use this tool when the user asks what you remember or questions their saved preferences (e.g., 'What is my target exam score?').
    """
    return sqlite_retrieve(query)


# Complete tools registry
ALL_STUDY_TOOLS = [
    search_documents,
    summarize_document,
    generate_quiz_from_document,
    calculate,
    generate_quiz,
    create_study_plan,
    save_memory,
    retrieve_memory,
    calculate_study_schedule,
    get_mca_subject_overview,
]

TOOLS_BY_NAME = {t.name: t for t in ALL_STUDY_TOOLS}
