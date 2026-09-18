"""
LangGraph Workflow Assembly and Compiler for SmartStudy AI.
Constructs the StateGraph with agent reasoning, tool execution cycles,
conditional routing, SQLite/Memory checkpointing, and execution wrappers.
"""

import sqlite3
from typing import Optional, Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END

from graph.state import StudyAgentState
from graph.nodes import agent_node, safe_tool_executor, final_response_node, should_continue
from config.settings import DEFAULT_MODEL, DEFAULT_OLLAMA_HOST, GRAPH_CHECKPOINTS_PATH, MEMORY_DIR


def get_checkpointer():
    """
    Initialize a persistent checkpointer.
    Attempts to use LangGraph's SqliteSaver with persistent db,
    falling back to MemorySaver if sqlite package is not installed.
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(GRAPH_CHECKPOINTS_PATH), check_same_thread=False)
        return SqliteSaver(conn)
    except (ImportError, Exception):
        try:
            from langgraph.checkpoint.memory import MemorySaver
            return MemorySaver()
        except ImportError:
            return None


def create_study_agent_graph(checkpointer: Optional[Any] = None):
    """
    Build and compile the LangGraph StateGraph workflow.

    Workflow topology:
    START ──► agent ──► (should_continue?)
                         ├── 'tools' ──► tools ──► agent
                         └── 'final_response' ──► final_response ──► END
    """
    workflow = StateGraph(StudyAgentState)

    # Add Nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", safe_tool_executor)
    workflow.add_node("final_response", final_response_node)

    # Entry point
    workflow.add_edge(START, "agent")

    # Conditional router from agent
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "final_response": "final_response",
        }
    )

    # Tool cycle: return to agent to synthesize results
    workflow.add_edge("tools", "agent")

    # Exit edge
    workflow.add_edge("final_response", END)

    # Compile with optional checkpointer
    active_checkpointer = checkpointer if checkpointer is not None else get_checkpointer()
    return workflow.compile(checkpointer=active_checkpointer)


# Singleton compiled graph instance
study_graph_app = create_study_agent_graph()


def run_study_agent(
    user_query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    thread_id: str = "mca_student_session",
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_OLLAMA_HOST,
) -> Dict[str, Any]:
    """
    Execute the compiled LangGraph workflow for a student interaction.

    Args:
        user_query: The current question or command.
        conversation_history: Past user and assistant turns.
        thread_id: Session identifier for SQLite checkpointer.
        model: Target Ollama model name.
        base_url: Ollama server endpoint.

    Returns:
        Dictionary containing final_response, tool_results, retrieved_documents,
        and debug_trace audit entries.
    """
    history = conversation_history or []
    initial_messages: List[BaseMessage] = []

    for turn in history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role == "user":
            initial_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            initial_messages.append(AIMessage(content=content))

    # Append current turn
    initial_messages.append(HumanMessage(content=user_query))

    initial_state: StudyAgentState = {
        "messages": initial_messages,
        "user_message": user_query,
        "selected_tools": [],
        "retrieved_documents": [],
        "tool_results": [],
        "final_response": "",
        "debug_trace": [],
        "error": None,
    }

    config = {
        "configurable": {
            "thread_id": thread_id,
            "model": model,
            "base_url": base_url,
        }
    }

    final_state = study_graph_app.invoke(initial_state, config=config)

    # Extract final text
    final_resp = final_state.get("final_response", "")
    if not final_resp:
        for m in reversed(final_state.get("messages", [])):
            if isinstance(m, AIMessage) and m.content:
                final_resp = m.content
                break

    return {
        "final_response": final_resp,
        "selected_tools": final_state.get("selected_tools", []),
        "tool_results": final_state.get("tool_results", []),
        "retrieved_documents": final_state.get("retrieved_documents", []),
        "debug_trace": final_state.get("debug_trace", []),
        "error": final_state.get("error"),
    }
