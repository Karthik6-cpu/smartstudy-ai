"""
Structured Quiz Generator for SmartStudy AI.
Generates interactive multiple-choice questions (MCQs) grounded in MCA topics
or specific uploaded PDF documents, with reliable JSON parsing and fallbacks.
"""

import json
import re
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from llm.langchain_client import get_chat_ollama, LangChainOllamaFactory
from rag.retriever import RAGRetriever
from rag.vector_store import VectorStore
from config.settings import DEFAULT_MODEL, DEFAULT_OLLAMA_HOST


QUIZ_SYSTEM_PROMPT = """You are an MCA Computer Science exam creator.
Your task is to generate structured Multiple Choice Questions (MCQs) strictly in JSON format.
Each question must have exactly 4 options labeled A, B, C, D, a correct_answer single letter ('A', 'B', 'C', or 'D'), and a detailed explanation.

Output ONLY a raw JSON array matching this schema:
[
  {
    "id": 1,
    "question": "What is the primary characteristic of ...?",
    "options": [
      "A) Option description",
      "B) Option description",
      "C) Option description",
      "D) Option description"
    ],
    "correct_answer": "B",
    "explanation": "Detailed explanation of why B is correct."
  }
]
Do not include any conversational preamble or postscript. Output valid JSON only."""


def parse_quiz_json(raw_text: str) -> Optional[List[Dict[str, Any]]]:
    """Extract and validate JSON question array from model output."""
    if not raw_text:
        return None

    # Try direct parse
    try:
        data = json.loads(raw_text.strip())
        if isinstance(data, list) and len(data) > 0 and "question" in data[0]:
            return data
    except Exception:
        pass

    # Try regex extraction from ```json ... ``` or [ ... ]
    match = re.search(r"\[\s*\{.*\}\s*\]", raw_text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list) and len(data) > 0:
                return data
        except Exception:
            pass

    return None


def get_curated_fallback_questions(topic: str, count: int = 5) -> List[Dict[str, Any]]:
    """Curated, high-yield MCA fallback questions for DSA, DBMS, OS, Networks, and SE."""
    t_lower = topic.lower()

    if "cmmi" in t_lower or "software" in t_lower or "se" in t_lower:
        bank = [
            {
                "id": 1,
                "question": "In CMMI, what characterizes Maturity Level 2 (Managed)?",
                "options": [
                    "A) Processes are unpredictable and reactive",
                    "B) Processes are planned, performed, measured, and controlled at the project level",
                    "C) Organization-wide standard processes are documented and proactive",
                    "D) Processes are controlled using statistical techniques",
                ],
                "correct_answer": "B",
                "explanation": "At Level 2 (Managed), projects ensure that processes are planned, executed, measured, and controlled.",
            },
            {
                "id": 2,
                "question": "How many maturity levels are defined in the CMMI Staged Representation?",
                "options": ["A) 3 Levels", "B) 4 Levels", "C) 5 Levels", "D) 6 Levels"],
                "correct_answer": "C",
                "explanation": "CMMI staged model defines 5 maturity levels: Initial, Managed, Defined, Quantitatively Managed, and Optimizing.",
            },
            {
                "id": 3,
                "question": "Which CMMI level focuses on continuous process improvement and technological innovation?",
                "options": ["A) Level 2 - Managed", "B) Level 3 - Defined", "C) Level 4 - Quantitatively Managed", "D) Level 5 - Optimizing"],
                "correct_answer": "D",
                "explanation": "Level 5 (Optimizing) is dedicated to continual process improvement through incremental and innovative technological changes.",
            },
            {
                "id": 4,
                "question": "What is the primary difference between Staged and Continuous representations in CMMI?",
                "options": [
                    "A) Staged uses maturity levels; Continuous uses capability levels for individual process areas",
                    "B) Staged is only for hardware; Continuous is for software",
                    "C) Continuous does not require documentation",
                    "D) Staged representation has only 3 levels",
                ],
                "correct_answer": "A",
                "explanation": "Staged representation provides a predefined roadmap using maturity levels, whereas Continuous allows organizations to select and improve specific process areas using capability levels.",
            },
            {
                "id": 5,
                "question": "Which SDLC model incorporates risk analysis in every cycle of development?",
                "options": ["A) Waterfall Model", "B) Spiral Model", "C) Rapid Application Development (RAD)", "D) V-Model"],
                "correct_answer": "B",
                "explanation": "The Spiral Model combines the iterative nature of prototyping with the controlled aspects of the Waterfall model, featuring explicit risk assessment in every spiral quadrant.",
            },
        ]
    elif "dbms" in t_lower or "join" in t_lower or "sql" in t_lower or "normal" in t_lower:
        bank = [
            {
                "id": 1,
                "question": "Which SQL JOIN returns all rows from the left table, and matching rows from the right table?",
                "options": ["A) INNER JOIN", "B) LEFT OUTER JOIN", "C) RIGHT OUTER JOIN", "D) CROSS JOIN"],
                "correct_answer": "B",
                "explanation": "LEFT OUTER JOIN returns all tuples from the left relation, filling missing attributes from the right relation with NULLs.",
            },
            {
                "id": 2,
                "question": "A table is in Second Normal Form (2NF) if and only if it is in 1NF and:",
                "options": [
                    "A) Has no transitive dependencies",
                    "B) Every non-prime attribute is fully functionally dependent on every candidate key",
                    "C) Has no multi-valued dependencies",
                    "D) Has a composite primary key",
                ],
                "correct_answer": "B",
                "explanation": "2NF eliminates partial functional dependency, meaning all non-key attributes must depend on the whole candidate key.",
            },
            {
                "id": 3,
                "question": "In transaction management, which property guarantees that either all operations of a transaction succeed or none do?",
                "options": ["A) Atomicity", "B) Consistency", "C) Isolation", "D) Durability"],
                "correct_answer": "A",
                "explanation": "Atomicity (the 'A' in ACID) ensures the 'all-or-nothing' execution of database transactions.",
            },
            {
                "id": 4,
                "question": "What is the result of a NATURAL JOIN on two relations that have no common attributes?",
                "options": ["A) Empty relation", "B) Cartesian Product (Cross Join)", "C) Syntax Error", "D) Left Outer Join"],
                "correct_answer": "B",
                "explanation": "When two relations share no common attribute names, a natural join reduces to a Cartesian (Cross) Product.",
            },
            {
                "id": 5,
                "question": "Which database indexing structure is most commonly used for range-based queries?",
                "options": ["A) Hash Index", "B) B+ Tree", "C) Binary Search Tree", "D) Bitmap Index"],
                "correct_answer": "B",
                "explanation": "B+ Trees store all records in linked leaf nodes, making sequential range scans and boundary traversals extremely fast (O(log N)).",
            },
        ]
    elif "os" in t_lower or "deadlock" in t_lower or "thread" in t_lower or "schedul" in t_lower:
        bank = [
            {
                "id": 1,
                "question": "Which of the following is NOT one of Coffman's four conditions required for Deadlock?",
                "options": ["A) Mutual Exclusion", "B) Hold and Wait", "C) Preemption Permitted", "D) Circular Wait"],
                "correct_answer": "C",
                "explanation": "Deadlock requires NO PREEMPTION. If preemption is permitted, resources can be reclaimed to break deadlock.",
            },
            {
                "id": 2,
                "question": "Which CPU scheduling algorithm is optimal in terms of minimizing average waiting time?",
                "options": ["A) First-Come, First-Served (FCFS)", "B) Shortest Job First (SJF)", "C) Round Robin", "D) Priority Scheduling"],
                "correct_answer": "B",
                "explanation": "Shortest Job First (SJF) is provably optimal for minimizing average waiting time among non-preemptive scheduling algorithms.",
            },
            {
                "id": 3,
                "question": "What happens during a Thread context switch compared to a Process context switch?",
                "options": [
                    "A) Memory page tables must be completely reloaded",
                    "B) It is faster because threads share the same virtual address space and memory",
                    "C) It requires disk swapping",
                    "D) Thread context switching is not supported by modern operating systems",
                ],
                "correct_answer": "B",
                "explanation": "Threads in the same process share code, data, and address space. Switching between them does not invalidate TLB or reload page tables.",
            },
            {
                "id": 4,
                "question": "The Banker's Algorithm is used for:",
                "options": ["A) Deadlock Prevention", "B) Deadlock Avoidance", "C) Deadlock Detection", "D) Memory Swapping"],
                "correct_answer": "B",
                "explanation": "Banker's Algorithm tests for safe states before granting resource allocation requests to avoid entering unsafe deadlock states.",
            },
            {
                "id": 5,
                "question": "Belady's Anomaly describes a phenomenon where:",
                "options": [
                    "A) Increasing page frames results in more page faults under FIFO",
                    "B) Processes starve under Round Robin",
                    "C) CPU utilization drops to 0%",
                    "D) Virtual memory exceeds physical swap space",
                ],
                "correct_answer": "A",
                "explanation": "Belady's Anomaly occurs in FIFO page replacement, where adding more physical page frames unexpectedly increases the number of page faults.",
            },
        ]
    else:  # DSA default
        bank = [
            {
                "id": 1,
                "question": "What is the worst-case time complexity of standard Binary Search?",
                "options": ["A) O(1)", "B) O(log N)", "C) O(N)", "D) O(N log N)"],
                "correct_answer": "B",
                "explanation": "Binary search divides the search space in half at each iteration, yielding logarithmic time O(log N).",
            },
            {
                "id": 2,
                "question": "Which data structure follows the Last-In, First-Out (LIFO) principle and is used for recursion call stacks?",
                "options": ["A) Queue", "B) Stack", "C) Linked List", "D) Priority Queue"],
                "correct_answer": "B",
                "explanation": "Stacks operate on LIFO, making them ideal for function call management, backtracking, and expression parsing.",
            },
            {
                "id": 3,
                "question": "What is the balance factor allowed for any node in an AVL Tree?",
                "options": ["A) Exactly 0", "B) {-1, 0, +1}", "C) {-2, 0, +2}", "D) Any positive integer"],
                "correct_answer": "B",
                "explanation": "An AVL tree is a self-balancing binary search tree where the height difference (balance factor) between left and right subtrees must be -1, 0, or +1.",
            },
            {
                "id": 4,
                "question": "Which graph algorithm finds the single-source shortest paths in a graph with non-negative edge weights?",
                "options": ["A) Prim's Algorithm", "B) Dijkstra's Algorithm", "C) Kruskal's Algorithm", "D) Floyd-Warshall Algorithm"],
                "correct_answer": "B",
                "explanation": "Dijkstra's greedy algorithm finds the shortest path from a starting vertex to all other vertices in non-negative weighted graphs.",
            },
            {
                "id": 5,
                "question": "What is the time complexity to find an element in a Hash Table in the average case?",
                "options": ["A) O(1)", "B) O(log N)", "C) O(N)", "D) O(N^2)"],
                "correct_answer": "A",
                "explanation": "Under good hash functions with uniform distribution, average lookup, insertion, and deletion time is O(1).",
            },
        ]

    return bank[:count]


def generate_structured_quiz(
    topic: str,
    num_questions: int = 5,
    difficulty: str = "Intermediate",
    document_name: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_OLLAMA_HOST,
) -> List[Dict[str, Any]]:
    """
    Generate an interactive multiple-choice quiz on an MCA topic or specific uploaded document.
    """
    context_str = ""

    # 1. If document is specified, retrieve excerpts to ground the quiz in the user's PDF
    if document_name and document_name != "None (General Knowledge)":
        retriever = RAGRetriever()
        docs = retriever.retrieve_documents(f"{topic} in {document_name}")
        # Filter chunks by document name if possible
        doc_chunks = [d for d in docs if d.metadata.get("source") == document_name] or docs
        if doc_chunks:
            context_str = "\n\n".join(f"[Excerpt {i+1}]: {d.page_content}" for i, d in enumerate(doc_chunks[:4]))

    user_prompt = (
        f"Topic: {topic}\n"
        f"Difficulty: {difficulty}\n"
        f"Number of Questions: {num_questions}\n"
    )
    if context_str:
        user_prompt += (
            f"\nGround the questions strictly in this uploaded course document excerpt:\n"
            f"Document: {document_name}\n"
            f"{context_str}\n"
        )
    else:
        user_prompt += "Generate comprehensive questions covering core MCA curriculum concepts on this topic."

    # 2. Attempt generation with Ollama
    is_ok, _ = LangChainOllamaFactory.validate_connection(base_url)
    if is_ok:
        try:
            llm = get_chat_ollama(model=model, base_url=base_url, temperature=0.3)
            response = llm.invoke([
                SystemMessage(content=QUIZ_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ])
            parsed = parse_quiz_json(response.content)
            if parsed and len(parsed) >= 1:
                # Ensure each question has required keys
                clean_list = []
                for i, q in enumerate(parsed[:num_questions], start=1):
                    q["id"] = i
                    if "options" in q and len(q["options"]) == 4 and "correct_answer" in q:
                        clean_list.append(q)
                if len(clean_list) >= 1:
                    return clean_list
        except Exception:
            pass

    # 3. Fallback to curated question bank if LLM fails or is offline
    return get_curated_fallback_questions(topic, count=num_questions)
