"""
Typed State Definition for SmartStudy AI LangGraph Agent.
Maintains user queries, conversation history, tool executions, retrieved documents,
and developer debug traces across graph nodes.
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class StudyAgentState(TypedDict):
    """
    Explicit Typed State for the SmartStudy AI LangGraph workflow.
    """
    # Conversation messages with automatic reducer append
    messages: Annotated[List[BaseMessage], add_messages]

    # Current user question/input
    user_message: str

    # Names of tools selected during agent reasoning
    selected_tools: List[str]

    # Document excerpts retrieved if search_documents was invoked
    retrieved_documents: List[Dict[str, Any]]

    # Structured tool execution outputs
    tool_results: List[Dict[str, Any]]

    # Final synthesized answer for the user
    final_response: str

    # Detailed audit trail for developer/debug mode
    debug_trace: List[Dict[str, Any]]

    # Error message if a tool or model failure occurred
    error: Optional[str]
