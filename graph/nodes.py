"""
Graph Nodes and Conditional Routing for SmartStudy AI LangGraph Agent.
Implements Agent reasoning node, Tool execution with error recovery,
and Final Response synthesis with debug tracing.
"""

from typing import Dict, Any, List, Optional
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
)
from langgraph.prebuilt import ToolNode

from graph.state import StudyAgentState
from tools.study_tools import ALL_STUDY_TOOLS, TOOLS_BY_NAME
from llm.langchain_client import get_chat_ollama, LangChainOllamaFactory
from config.settings import AGENT_SYSTEM_PROMPT, DEFAULT_MODEL, DEFAULT_OLLAMA_HOST
from chains.agent import StudyAgent


def get_agent_llm(config: Optional[Dict[str, Any]] = None):
    """Instantiate configured ChatOllama with tools bound."""
    cfg = config or {}
    configurable = cfg.get("configurable", {})
    model = configurable.get("model", DEFAULT_MODEL)
    base_url = configurable.get("base_url", DEFAULT_OLLAMA_HOST)
    temperature = configurable.get("temperature", 0.2)

    llm = get_chat_ollama(model=model, base_url=base_url, temperature=temperature)
    return llm.bind_tools(ALL_STUDY_TOOLS)


def agent_node(state: StudyAgentState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Agent Reasoning Node.
    Calls ChatOllama with bound tools to determine if tools are needed or answers directly.
    Includes heuristic intent fallback for local Ollama models with incomplete function-calling support.
    """
    messages = list(state.get("messages", []))
    user_msg = state.get("user_message", "")
    debug_trace = list(state.get("debug_trace", []))

    # Ensure system prompt is present at the start
    if not messages or not isinstance(messages[0], SystemMessage):
        messages.insert(0, SystemMessage(content=AGENT_SYSTEM_PROMPT))

    try:
        llm_with_tools = get_agent_llm(config)
        response = llm_with_tools.invoke(messages)
    except Exception as e:
        err_msg = f"Failed to invoke LLM in agent_node: {str(e)}"
        response = AIMessage(content=f"⚠️ I encountered an error connecting to the model: {str(e)}")
        return {
            "messages": [response],
            "selected_tools": [],
            "error": err_msg,
            "debug_trace": debug_trace + [{"node": "agent", "error": err_msg}]
        }

    tool_calls = getattr(response, "tool_calls", [])

    # Heuristic fallback if local model outputs text instead of structured tool_calls
    if not tool_calls:
        helper = StudyAgent()
        fallback = helper.detect_intent_fallback(user_msg)
        if fallback:
            tname, targs = fallback
            tool_calls = [{"name": tname, "args": targs, "id": f"call_{tname}"}]
            # Attach synthetic tool calls to AIMessage so LangGraph router picks it up
            response.tool_calls = tool_calls

    selected_tools = [tc.get("name") for tc in tool_calls]

    trace_entry = {
        "node": "agent",
        "user_question": user_msg,
        "selected_tools": selected_tools,
        "tool_calls": tool_calls,
        "has_tool_calls": bool(tool_calls),
    }

    return {
        "messages": [response],
        "selected_tools": selected_tools,
        "debug_trace": debug_trace + [trace_entry],
    }


def safe_tool_executor(state: StudyAgentState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Tool Execution Node with Error Containment.
    Executes tool calls requested by the agent, intercepts any errors gracefully,
    and returns informative ToolMessages without exposing raw stack traces.
    """
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])
    debug_trace = list(state.get("debug_trace", []))
    prior_tool_results = list(state.get("tool_results", []))
    prior_retrieved_docs = list(state.get("retrieved_documents", []))

    tool_messages: List[BaseMessage] = []
    current_results: List[Dict[str, Any]] = []

    for tc in tool_calls:
        tool_name = tc.get("name")
        tool_args = tc.get("args", {})
        tool_id = tc.get("id", f"call_{tool_name}")

        tool = TOOLS_BY_NAME.get(tool_name)
        if not tool:
            err_content = f"Notice: Tool '{tool_name}' is not recognized. Please provide an answer using general knowledge."
            tool_messages.append(ToolMessage(content=err_content, tool_call_id=tool_id, name=tool_name))
            current_results.append({"tool": tool_name, "args": tool_args, "result": err_content, "status": "not_found"})
            continue

        try:
            # Execute tool safely
            raw_output = tool.invoke(tool_args)
            output_str = str(raw_output)

            # If search_documents was executed, record in retrieved_documents for citations
            if tool_name == "search_documents":
                prior_retrieved_docs.append({"query": tool_args.get("query", ""), "result": output_str})

            tool_messages.append(ToolMessage(content=output_str, tool_call_id=tool_id, name=tool_name))
            current_results.append({"tool": tool_name, "args": tool_args, "result": output_str, "status": "success"})

        except Exception as e:
            # Graceful error recovery: inform the agent politely without raw tracebacks
            friendly_error = (
                f"Notice: The tool '{tool_name}' encountered an error while processing: {str(e)}. "
                "Please notify the student politely and answer to the best of your general ability."
            )
            tool_messages.append(ToolMessage(content=friendly_error, tool_call_id=tool_id, name=tool_name))
            current_results.append({"tool": tool_name, "args": tool_args, "result": friendly_error, "status": "error"})

    trace_entry = {
        "node": "tools",
        "tool_executions": current_results,
    }

    return {
        "messages": tool_messages,
        "tool_results": prior_tool_results + current_results,
        "retrieved_documents": prior_retrieved_docs,
        "debug_trace": debug_trace + [trace_entry],
    }


def final_response_node(state: StudyAgentState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Final Response Node.
    Extracts the synthesized answer from the agent conversation and finalizes the state.
    """
    last_msg = state["messages"][-1]
    content = getattr(last_msg, "content", "")
    debug_trace = list(state.get("debug_trace", []))

    trace_entry = {
        "node": "final_response",
        "final_answer": content,
    }

    return {
        "final_response": content,
        "debug_trace": debug_trace + [trace_entry],
    }


def should_continue(state: StudyAgentState) -> str:
    """
    Conditional Edge Router.
    Routes to 'tools' if the agent requested tool calls, otherwise to 'final_response'.
    """
    messages = state.get("messages", [])
    if not messages:
        return "final_response"

    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", [])

    if tool_calls:
        return "tools"
    return "final_response"
