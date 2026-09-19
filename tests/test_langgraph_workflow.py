"""
Automated Test Suite for LangGraph Agent Workflow in SmartStudy AI.
Tests StudyAgentState, Graph Compilation, Node Execution,
Safe Tool Containment, Conditional Routing, and Developer Debug Traces.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from graph.state import StudyAgentState
from graph.nodes import (
    agent_node,
    safe_tool_executor,
    final_response_node,
    should_continue,
)
from graph.study_graph import create_study_agent_graph, run_study_agent


def test_study_agent_state_schema():
    """Verify StudyAgentState supports all expected state attributes."""
    print("Testing StudyAgentState schema...")
    test_state: StudyAgentState = {
        "messages": [HumanMessage(content="Explain Binary Search")],
        "user_message": "Explain Binary Search",
        "selected_tools": ["search_documents"],
        "retrieved_documents": [{"source": "notes.pdf", "page": 1}],
        "tool_results": [{"tool": "search_documents", "result": "Sample excerpt"}],
        "final_response": "Binary search is an O(log N) algorithm...",
        "debug_trace": [{"node": "agent", "decision": "tools"}],
        "error": None,
    }

    assert len(test_state["messages"]) == 1
    assert test_state["user_message"] == "Explain Binary Search"
    assert test_state["selected_tools"] == ["search_documents"]
    assert len(test_state["tool_results"]) == 1
    print(" StudyAgentState schema validated.")


def test_conditional_routing():
    """Verify should_continue routes to 'tools' when tool_calls exist, and 'final_response' otherwise."""
    print("Testing conditional routing logic...")

    # 1. State with tool calls -> should route to 'tools'
    ai_with_tools = AIMessage(
        content="",
        tool_calls=[{"name": "calculate", "args": {"expression": "25% of 480"}, "id": "call_calc_1"}]
    )
    state_tools: StudyAgentState = {
        "messages": [HumanMessage(content="Calculate 25% of 480"), ai_with_tools],
        "user_message": "Calculate 25% of 480",
        "selected_tools": ["calculate"],
        "retrieved_documents": [],
        "tool_results": [],
        "final_response": "",
        "debug_trace": [],
        "error": None,
    }
    route_tools = should_continue(state_tools)
    assert route_tools == "tools", f"Expected 'tools', got '{route_tools}'"

    # 2. State without tool calls -> should route to 'final_response'
    ai_direct = AIMessage(content="Binary Search repeatedly divides the search interval in half.")
    state_direct: StudyAgentState = {
        "messages": [HumanMessage(content="Explain Binary Search"), ai_direct],
        "user_message": "Explain Binary Search",
        "selected_tools": [],
        "retrieved_documents": [],
        "tool_results": [],
        "final_response": "",
        "debug_trace": [],
        "error": None,
    }
    route_direct = should_continue(state_direct)
    assert route_direct == "final_response", f"Expected 'final_response', got '{route_direct}'"

    print(" Conditional routing validated.")


def test_safe_tool_executor_success_and_recovery():
    """Verify Tool execution node runs tools, updates results, and recovers gracefully from errors."""
    print("Testing ToolNode execution and error recovery...")

    # 1. Successful tool execution (calculate)
    ai_call = AIMessage(
        content="",
        tool_calls=[{"name": "calculate", "args": {"expression": "25% of 480"}, "id": "call_1"}]
    )
    state: StudyAgentState = {
        "messages": [HumanMessage(content="Calculate 25% of 480"), ai_call],
        "user_message": "Calculate 25% of 480",
        "selected_tools": ["calculate"],
        "retrieved_documents": [],
        "tool_results": [],
        "final_response": "",
        "debug_trace": [],
        "error": None,
    }

    result = safe_tool_executor(state)
    assert len(result["messages"]) == 1
    tool_msg = result["messages"][0]
    assert isinstance(tool_msg, ToolMessage)
    assert "Result: 120" in tool_msg.content
    assert result["tool_results"][0]["status"] == "success"

    # 2. Error recovery (unknown tool or failing parameter) - must NOT crash
    ai_bad_call = AIMessage(
        content="",
        tool_calls=[{"name": "non_existent_tool", "args": {}, "id": "call_err"}]
    )
    state_bad: StudyAgentState = {
        "messages": [HumanMessage(content="Run fake tool"), ai_bad_call],
        "user_message": "Run fake tool",
        "selected_tools": ["non_existent_tool"],
        "retrieved_documents": [],
        "tool_results": [],
        "final_response": "",
        "debug_trace": [],
        "error": None,
    }

    bad_result = safe_tool_executor(state_bad)
    assert len(bad_result["messages"]) == 1
    assert "Notice: Tool 'non_existent_tool' is not recognized" in bad_result["messages"][0].content
    assert bad_result["tool_results"][0]["status"] == "not_found"

    print(" ToolNode execution and error containment validated.")


def test_final_response_node():
    """Verify final_response_node extracts final text and records debug trace."""
    print("Testing final_response_node...")
    ai_final = AIMessage(content="Final synthesized explanation for the student.")
    state: StudyAgentState = {
        "messages": [HumanMessage(content="Question"), ai_final],
        "user_message": "Question",
        "selected_tools": [],
        "retrieved_documents": [],
        "tool_results": [],
        "final_response": "",
        "debug_trace": [],
        "error": None,
    }

    out = final_response_node(state)
    assert out["final_response"] == "Final synthesized explanation for the student."
    assert len(out["debug_trace"]) == 1
    assert out["debug_trace"][0]["node"] == "final_response"

    print(" final_response_node validated.")


def test_graph_compilation():
    """Verify LangGraph StateGraph compiles and registers all expected nodes."""
    print("Testing LangGraph StateGraph compilation...")
    graph = create_study_agent_graph()
    assert graph is not None

    # Verify node names in the compiled graph
    node_keys = graph.nodes.keys()
    assert "agent" in node_keys
    assert "tools" in node_keys
    assert "final_response" in node_keys

    print(" LangGraph StateGraph compiled successfully.")


if __name__ == "__main__":
    test_study_agent_state_schema()
    test_conditional_routing()
    test_safe_tool_executor_success_and_recovery()
    test_final_response_node()
    test_graph_compilation()
    print("\nAll LangGraph workflow tests passed successfully! [SUCCESS]")
