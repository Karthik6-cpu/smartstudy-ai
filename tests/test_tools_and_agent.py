"""
Automated Test Suite for LangChain Tool Calling in SmartStudy AI.
Tests Safe Calculator, SQLite Memory, Tool Definitions, and Agent Intent Routing.
"""

import sys
import os
import shutil
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.safe_calculator import safe_calculate
from memory.sqlite_memory import (
    save_memory,
    retrieve_memory,
    get_memory,
    list_all_memories,
    delete_memory,
    clear_all_memories,
)
from tools.study_tools import (
    search_documents,
    calculate,
    generate_quiz,
    create_study_plan,
    save_memory as tool_save_memory,
    retrieve_memory as tool_retrieve_memory,
    ALL_STUDY_TOOLS,
)
from chains.agent import StudyAgent


def test_safe_calculator():
    """Verify safe calculator handles math, percentages, functions, and rejects malicious payloads."""
    print("Testing Safe Calculator...")

    # 1. Standard Arithmetic
    r1 = safe_calculate("5 + 3 * 2")
    assert r1 == "Result: 11", f"Expected 11, got {r1}"

    # 2. Percentage calculation
    r2 = safe_calculate("25% of 480")
    assert r2 == "Result: 120", f"Expected 120, got {r2}"

    # 3. Math functions & powers
    r3 = safe_calculate("sqrt(144) + 2^8")
    assert r3 == "Result: 268", f"Expected 268, got {r3}"

    # 4. Zero division safety
    r4 = safe_calculate("100 / 0")
    assert "Division by zero" in r4

    # 5. Security: reject arbitrary python code
    r5 = safe_calculate("__import__('os').system('dir')")
    assert "Security Error" in r5 or "Syntax Error" in r5

    r6 = safe_calculate("open('secrets.txt', 'r')")
    assert "Security Error" in r6

    print(" Safe Calculator passed all checks.")


def test_sqlite_memory():
    """Verify SQLite memory CRUD operations."""
    print("Testing SQLite Memory...")

    # Clear prior test state
    clear_all_memories()

    # Save memory
    res_save = save_memory("target_score", "90% in End-Sem Finals")
    assert "Successfully saved" in res_save

    # Get memory
    val = get_memory("target_score")
    assert val == "90% in End-Sem Finals"

    # Search memory
    ret_text = retrieve_memory("score")
    assert "target_score" in ret_text
    assert "90%" in ret_text

    # List all
    all_m = list_all_memories()
    assert len(all_m) == 1
    assert all_m[0]["key"] == "target_score"

    # Delete memory
    del_ok = delete_memory("target_score")
    assert del_ok is True
    assert get_memory("target_score") is None

    print(" SQLite Memory passed all checks.")


def test_langchain_tool_definitions():
    """Verify all 5 LangChain tools can be invoked and return valid strings."""
    assert len(ALL_STUDY_TOOLS) >= 6

    # 1. calculate tool
    calc_out = calculate.invoke({"expression": "25% of 480"})
    assert calc_out == "Result: 120"

    # 2. memory tools
    tool_save_memory.invoke({"key": "favorite_subject", "value": "Operating Systems"})
    mem_out = tool_retrieve_memory.invoke({"query": "favorite_subject"})
    assert "Operating Systems" in mem_out

    # 3. search_documents tool
    doc_out = search_documents.invoke({"query": "CMMI"})
    assert isinstance(doc_out, str)

    # 4. generate_quiz tool
    quiz_out = generate_quiz.invoke({"topic": "DBMS joins", "number_of_questions": 3, "difficulty": "Beginner"})
    assert isinstance(quiz_out, str)
    assert len(quiz_out) > 0

    # 5. create_study_plan tool
    plan_out = create_study_plan.invoke({"subjects": "DSA, OS", "available_hours": 3.0, "exam_date": "in 4 weeks"})
    assert isinstance(plan_out, str)
    assert len(plan_out) > 0

    print(" LangChain Tools validated successfully.")


def test_agent_intent_fallback():
    """Verify autonomous intent detection for key user requests."""
    print("Testing Agent Intent Routing Fallback...")
    agent = StudyAgent()

    # 1. Math query
    intent1 = agent.detect_intent_fallback("Calculate 25% of 480.")
    assert intent1 is not None
    assert intent1[0] == "calculate"

    # 2. Uploaded notes search query
    intent2 = agent.detect_intent_fallback("Explain CMMI from my uploaded Software Engineering notes.")
    assert intent2 is not None
    assert intent2[0] == "search_documents"

    # 3. Quiz request
    intent3 = agent.detect_intent_fallback("Give me 10 MCQs about DBMS joins.")
    assert intent3 is not None
    assert intent3[0] == "generate_quiz"

    # 4. Remember memory
    intent4 = agent.detect_intent_fallback("Remember that my target exam score is 90%.")
    assert intent4 is not None
    assert intent4[0] == "save_memory"

    # 5. Retrieve memory
    intent5 = agent.detect_intent_fallback("What is my target exam score?")
    assert intent5 is not None
    assert intent5[0] == "retrieve_memory"

    # 6. General query (No tool needed)
    intent6 = agent.detect_intent_fallback("Explain CMMI in simple terms.")
    assert intent6 is None, "General question should not force a tool"

    print(" Agent Intent Routing passed all checks.")


if __name__ == "__main__":
    test_safe_calculator()
    test_sqlite_memory()
    test_langchain_tool_definitions()
    test_agent_intent_fallback()
    print("\nAll Tool Calling & Agent tests passed successfully! [SUCCESS]")
