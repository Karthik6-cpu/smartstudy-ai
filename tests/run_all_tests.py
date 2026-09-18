"""
Master Production Test Runner for SmartStudy AI.
Executes end-to-end verification across:
1. Configuration & Offline Error Handling
2. RAG Pipeline (PDF -> Chunks -> Embeddings -> FAISS -> Retrieval -> Deletion)
3. LangChain Prompt Templates, Retriever & LCEL Chains
4. Safe AST Calculator, SQLite Memory & Agent Intent Routing
5. Full Features (Quiz Scoring, Study Planner State Machine, Analytics)
6. LangGraph StateGraph (State Schema, Nodes, Routing, Error Containment)
"""

import sys
import os
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import all individual test suites
from tests.test_ollama import (
    test_config_settings,
    test_ollama_service_init,
    test_ollama_graceful_offline_handling,
)
from tests.test_rag_pipeline import (
    test_pdf_extraction_and_chunking,
    test_pdf_error_handling,
    test_vector_store_persistence_and_retrieval,
)
from tests.test_langchain_integration import (
    test_prompt_templates,
    test_langchain_retriever_interface,
    test_study_tools,
    test_lcel_chains_construction,
)
from tests.test_tools_and_agent import (
    test_safe_calculator,
    test_sqlite_memory,
    test_langchain_tool_definitions,
    test_agent_intent_fallback,
)
from tests.test_full_features import (
    test_quiz_storage_and_stats,
    test_study_planner_tasks_and_analytics,
    test_activity_logging,
    test_quiz_generator_parsing_and_fallbacks,
)
from tests.test_langgraph_workflow import (
    test_study_agent_state_schema,
    test_conditional_routing,
    test_safe_tool_executor_success_and_recovery,
    test_final_response_node,
    test_graph_compilation,
)


def run_test_module(module_name: str, test_functions: list) -> tuple[int, int]:
    """Execute a group of test functions and report results."""
    print(f"\n{'='*70}")
    print(f"▶ RUNNING SUITE: {module_name}")
    print(f"{'='*70}")

    passed = 0
    failed = 0

    for test_fn in test_functions:
        fn_name = test_fn.__name__
        try:
            start_t = time.time()
            test_fn()
            dur = (time.time() - start_t) * 1000
            print(f"  PASSED: {fn_name} ({dur:.1f}ms)")
            passed += 1
        except Exception as e:
            print(f"  FAILED: {fn_name} -> {str(e)}")
            failed += 1

    return passed, failed


def main():
    print(f"\n{'#'*70}")
    print("  SMARTSTUDY AI - PRODUCTION VERIFICATION SCORECARD")
    print(f"{'#'*70}")

    suites = [
        ("1. Configuration & Offline Resilience", [
            test_config_settings,
            test_ollama_service_init,
            test_ollama_graceful_offline_handling,
        ]),
        ("2. Local PDF RAG & FAISS Vector Store", [
            test_pdf_extraction_and_chunking,
            test_pdf_error_handling,
            test_vector_store_persistence_and_retrieval,
        ]),
        ("3. LangChain Prompts, Retriever & LCEL", [
            test_prompt_templates,
            test_langchain_retriever_interface,
            test_study_tools,
            test_lcel_chains_construction,
        ]),
        ("4. Safe Calculator, SQLite Memory & Agent Tools", [
            test_safe_calculator,
            test_sqlite_memory,
            test_langchain_tool_definitions,
            test_agent_intent_fallback,
        ]),
        ("5. Interactive Quizzes, Planner & Analytics", [
            test_quiz_storage_and_stats,
            test_study_planner_tasks_and_analytics,
            test_activity_logging,
            test_quiz_generator_parsing_and_fallbacks,
        ]),
        ("6. LangGraph StateGraph & Tool Containment", [
            test_study_agent_state_schema,
            test_conditional_routing,
            test_safe_tool_executor_success_and_recovery,
            test_final_response_node,
            test_graph_compilation,
        ]),
    ]

    total_passed = 0
    total_failed = 0

    overall_start = time.time()

    for name, fns in suites:
        p, f = run_test_module(name, fns)
        total_passed += p
        total_failed += f

    total_time = time.time() - overall_start

    print(f"\n{'='*70}")
    print("  FINAL PRODUCTION TEST RESULTS")
    print(f"{'='*70}")
    print(f"  Total Checks Executed : {total_passed + total_failed}")
    print(f"  Passed Checks         : {total_passed} ")
    print(f"  Failed Checks         : {total_failed} ")
    print(f"  Total Duration        : {total_time:.2f}s")
    print(f"{'='*70}")

    if total_failed == 0:
        print("🎉 ALL PRODUCTION INTEGRITY CHECKS PASSED WITH 100% SUCCESS!")
        return 0
    else:
        print(f"❌ {total_failed} check(s) failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
