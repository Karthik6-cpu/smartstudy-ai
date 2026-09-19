"""
Prompts package for SmartStudy AI.
Centralizes reusable LangChain ChatPromptTemplates.
"""

from prompts.templates import (
    GENERAL_STUDY_PROMPT,
    RAG_STUDY_PROMPT,
    QUIZ_GENERATION_PROMPT,
    STUDY_PLANNER_PROMPT,
)

__all__ = [
    "GENERAL_STUDY_PROMPT",
    "RAG_STUDY_PROMPT",
    "QUIZ_GENERATION_PROMPT",
    "STUDY_PLANNER_PROMPT",
]
