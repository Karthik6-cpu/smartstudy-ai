"""
Chains package for SmartStudy AI.
Defines modular LCEL (LangChain Expression Language) pipelines.
"""

from chains.study_chains import (
    create_general_study_chain,
    create_rag_chain,
    create_quiz_chain,
    create_planner_chain,
)

__all__ = [
    "create_general_study_chain",
    "create_rag_chain",
    "create_quiz_chain",
    "create_planner_chain",
]
