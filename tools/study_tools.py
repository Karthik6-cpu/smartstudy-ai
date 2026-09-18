"""
LangChain Tools for SmartStudy AI.
Implements the 5 requested agent tools:
1. search_documents(query)
2. calculate(expression)
3. generate_quiz(topic, number_of_questions, difficulty)
4. create_study_plan(subjects, available_hours, exam_date)
5. save_memory(key, value) & retrieve_memory(query)
"""

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
def search_documents(query: str) -> str:
    """
    Search the user's uploaded study material and lecture notes through the local RAG retriever.
    Use this tool when the user asks a question specifically referring to their uploaded notes,
    course syllabus, or lecture slides (e.g., 'Explain CMMI from my uploaded Software Engineering notes').
    """
    try:
        retriever = RAGRetriever()
        docs = retriever.retrieve_documents(query)
        if not docs:
            return (
                f"No relevant excerpts found in uploaded notes for query: '{query}'. "
                "The documents may not contain this topic, or no documents have been uploaded yet."
            )

        formatted_excerpts = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "Unknown Document")
            page = doc.metadata.get("page", "?")
            score = doc.metadata.get("score", "")
            snippet = doc.page_content.strip()
            formatted_excerpts.append(
                f"[Source {i}: {source} | Page {page} | Score: {score}]\n{snippet}"
            )

        return (
            f"Retrieved {len(docs)} relevant excerpt(s) from uploaded notes:\n\n"
            + "\n\n".join(formatted_excerpts)
        )
    except Exception as e:
        return f"Error executing document search: {str(e)}"


@tool
def calculate(expression: str) -> str:
    """
    Safely perform mathematical and arithmetic calculations (e.g., percentages, algebra, numbers).
    Use this tool whenever a mathematical calculation is required, such as 'Calculate 25% of 480',
    'sqrt(144) + 12', or complexity math.
    """
    return safe_calculate(expression)


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
    try:
        quiz_chain = create_quiz_chain()
        result = quiz_chain.invoke({
            "topic": topic,
            "difficulty": difficulty,
            "num_questions": str(number_of_questions),
        })
        return result
    except Exception as e:
        return (
            f"Generated Quiz Outline for '{topic}' ({number_of_questions} questions, {difficulty}):\n"
            f"- Question 1: Fundamental definition and characteristics of {topic}.\n"
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
        planner_chain = create_planner_chain()
        result = planner_chain.invoke({
            "subjects": subjects,
            "timeline": exam_date,
            "daily_hours": str(available_hours),
            "goal": "High retention and semester final exam success",
        })
        return result
    except Exception as e:
        sub_list = [s.strip() for s in subjects.split(",") if s.strip()]
        return (
            f"### 📅 Study Plan for {len(sub_list)} Subjects ({exam_date})\n"
            f"- **Daily Study Target:** {available_hours} hours/day\n"
            f"- **Subjects:** {', '.join(sub_list)}\n"
            f"- **Recommendation:** Allocate ~{available_hours / max(len(sub_list), 1):.1f} hours/day per subject.\n"
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
    calculate,
    generate_quiz,
    create_study_plan,
    save_memory,
    retrieve_memory,
]

TOOLS_BY_NAME = {t.name: t for t in ALL_STUDY_TOOLS}
