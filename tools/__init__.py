"""
Tools package for SmartStudy AI.
LangChain @tool definitions for document search, safe calculation,
quiz generation, study planning, and SQLite memory.
"""

from tools.study_tools import (
    search_documents,
    search_course_notes,
    calculate,
    calculate_study_schedule,
    get_mca_subject_overview,
    generate_quiz,
    create_study_plan,
    save_memory,
    retrieve_memory,
    ALL_STUDY_TOOLS,
    TOOLS_BY_NAME,
)

__all__ = [
    "search_documents",
    "search_course_notes",
    "calculate",
    "calculate_study_schedule",
    "get_mca_subject_overview",
    "generate_quiz",
    "create_study_plan",
    "save_memory",
    "retrieve_memory",
    "ALL_STUDY_TOOLS",
    "TOOLS_BY_NAME",
]
