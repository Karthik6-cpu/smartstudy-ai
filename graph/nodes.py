"""
Graph Nodes and Conditional Routing for SmartStudy AI LangGraph Agent.
Implements Agent reasoning node, Tool execution with error recovery,
and Final Response synthesis with debug tracing.
"""

import json
import re
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


def extract_tool_calls_from_content(content: str, user_query: str = "") -> List[Dict[str, Any]]:
    """
    Extract structured tool calls from raw JSON strings or markdown blocks emitted by local Ollama models.
    Supports:
    - {"type": "function", "function": {"name": "...", "parameters": {...}}}
    - {"name": "...", "parameters": {...}} or {"name": "...", "arguments": {...}}
    - [ {"name": "...", ...} ]
    - ```json ... ``` code blocks
    - Cleans up common schema hallucinations (e.g., duplicate 'topic': 'string').
    """
    if not content or "{" not in content:
        return []

    # Try extracting JSON code block or outermost JSON object
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if not match:
        match = re.search(r"(\{.*\})", content, re.DOTALL)

    if not match:
        return []

    raw_str = match.group(1).strip()

    # Detect real topic before any 'topic': 'string' schema collision
    topic_match = re.search(r'"topic"\s*:\s*"([^"]+)"', raw_str)
    real_topic = None
    if topic_match and topic_match.group(1).lower() not in ("string", "none", ""):
        real_topic = topic_match.group(1)

    try:
        data = json.loads(raw_str)
    except Exception:
        # Regex fallback for tool name and parameters
        name_m = re.search(r'"name"\s*:\s*"([a-zA-Z0-9_]+)"', raw_str)
        if name_m:
            tname = name_m.group(1)
            targs = {}
            if tname == "generate_quiz":
                targs = {"topic": real_topic or user_query or "DBMS joins", "number_of_questions": 5}
            elif tname == "calculate":
                expr_m = re.search(r'"expression"\s*:\s*"([^"]+)"', raw_str)
                targs = {"expression": expr_m.group(1) if expr_m else user_query}
            elif tname in ("search_documents", "retrieve_memory"):
                q_m = re.search(r'"query"\s*:\s*"([^"]+)"', raw_str)
                targs = {"query": q_m.group(1) if q_m else user_query}
            return [{"name": tname, "args": targs, "id": f"call_{tname}"}]
        return []

    tool_calls = []
    items = data if isinstance(data, list) else [data]

    for item in items:
        if not isinstance(item, dict):
            continue
        tname = None
        targs = {}
        if item.get("type") == "function" and "function" in item:
            fn = item["function"]
            tname = fn.get("name")
            targs = fn.get("parameters") or fn.get("arguments") or {}
        elif "name" in item:
            tname = item.get("name")
            targs = item.get("parameters") or item.get("arguments") or item.get("args") or {}

        if tname and tname in TOOLS_BY_NAME:
            if real_topic and str(targs.get("topic", "")).lower() == "string":
                targs["topic"] = real_topic

            if "number_of_questions" in targs:
                try:
                    targs["number_of_questions"] = int(targs["number_of_questions"])
                except Exception:
                    targs["number_of_questions"] = 5

            tool_calls.append({"name": tname, "args": targs, "id": f"call_{tname}"})

    return tool_calls


def check_conversational_greeting(query: str) -> Optional[str]:
    """
    Detect normal greetings and conversational pleasantries (ChatGPT-style).
    Returns an immediate, warm conversational response without triggering specialized tools.
    """
    if not query:
        return None
    q_clean = query.strip().lower().rstrip(".!?,;: ")
    
    # Exact and prefix greetings
    greetings = {
        "hello": "Hello! 👋 How can I help you with your studies today?",
        "hi": "Hi there! 👋 What topic would you like to explore or review today?",
        "hey": "Hey! 👋 Ready to help you with your MCA coursework, notes, or exam prep.",
        "good morning": "Good morning! ☀️ What would you like to study today?",
        "good afternoon": "Good afternoon! 📖 What topic are we reviewing today?",
        "good evening": "Good evening! 🌙 How can I assist your study session tonight?",
        "how are you": "I'm doing great, thank you! 😊 Ready to help you tackle your MCA concepts, revision, or questions. What are we studying today?",
        "how are you doing": "I'm doing great, thank you! 😊 How can I help you with your studies today?",
        "how's it going": "All good! 😊 Ready to assist with your studies whenever you are.",
        "how is it going": "Going great! 😊 What computer science topic are we working on?",
        "thank you": "You're welcome! 😊 Let me know whenever you'd like to explore more concepts or practice questions.",
        "thanks": "You're welcome! 😊 Feel free to ask more questions anytime.",
        "thank you so much": "Glad I could help! 😊 Let me know if you need anything else.",
        "thanks a lot": "You're very welcome! 😊 Keep up the great studying!",
        "goodbye": "Goodbye! 👋 Best of luck with your study session, and feel free to return whenever you need help!",
        "bye": "Goodbye! 👋 Have a productive study session!",
        "see you": "See you! 👋 Have a great time studying!",
    }
    
    for g_key, g_reply in greetings.items():
        if q_clean == g_key or q_clean.startswith(g_key + " ") or q_clean.endswith(" " + g_key):
            return g_reply
            
    return None


def agent_node(state: StudyAgentState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Agent Reasoning Node.
    Calls ChatOllama with bound tools to determine if tools are needed or answers directly.
    Supports ChatGPT-style conversational greetings, fast intent routing, and local model JSON extraction.
    """
    messages = list(state.get("messages", []))
    user_msg = state.get("user_message", "")
    debug_trace = list(state.get("debug_trace", []))

    # Always ensure user_msg is non-empty if human message exists
    if not user_msg:
        for m in reversed(messages):
            if isinstance(m, HumanMessage) and m.content:
                user_msg = m.content
                break

    # Check if tools have already been executed in the CURRENT turn (after the latest HumanMessage)
    last_human_idx = -1
    for idx, m in enumerate(messages):
        if isinstance(m, HumanMessage):
            last_human_idx = idx

    messages_in_turn = messages[last_human_idx:] if last_human_idx >= 0 else messages
    has_executed_tools = any(isinstance(m, ToolMessage) for m in messages_in_turn)

    # If tools have already executed and produced a direct structured result,
    # use the tool result directly without an unnecessary second slow LLM pass on CPU
    if has_executed_tools:
        tool_results = state.get("tool_results", [])
        if tool_results:
            last_tool = tool_results[-1].get("tool", "")
            last_res = tool_results[-1].get("result", "")
            if last_res:
                response = AIMessage(content=last_res)
                return {
                    "messages": [response],
                    "selected_tools": [],
                    "debug_trace": debug_trace + [{"node": "agent", "direct_tool_result": last_tool}],
                }

    # 1. CHATGPT-STYLE CONVERSATION: Instant friendly reply for normal greetings/pleasantries
    if not has_executed_tools and user_msg:
        greeting_reply = check_conversational_greeting(user_msg)
        if greeting_reply:
            response = AIMessage(content=greeting_reply)
            return {
                "messages": [response],
                "selected_tools": [],
                "debug_trace": debug_trace + [{"node": "agent", "intent": "conversation", "greeting": True}],
            }

    # 2. FAST INTENT ROUTING: If query matches a known tool intent (math, quiz, plan, memory, notes),
    # route immediately to tools without waiting 40-60s for heavy tool-binding inference on CPU!
    if not has_executed_tools and user_msg:
        helper = StudyAgent()
        fallback = helper.detect_intent_fallback(user_msg)
        if fallback:
            tname, targs = fallback
            tool_calls = [{"name": tname, "args": targs, "id": f"call_{tname}"}]
            response = AIMessage(content="", tool_calls=tool_calls)
            return {
                "messages": [response],
                "selected_tools": [tname],
                "debug_trace": debug_trace + [{"node": "agent", "fast_intent": tname, "tool_calls": tool_calls}],
            }

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

    # If model didn't return structured tool_calls, check if response.content contains raw JSON tool call
    if not tool_calls and response.content and "{" in response.content:
        extracted = extract_tool_calls_from_content(response.content, user_msg)
        if extracted:
            tool_calls = extracted
            response.tool_calls = extracted
            response.content = ""

    # If still no tool_calls, but content looks like raw function JSON, fallback to intent router
    if not tool_calls and response.content and any(k in response.content for k in ['"function"', '"name"', '"parameters"']):
        helper = StudyAgent()
        fallback = helper.detect_intent_fallback(user_msg)
        if fallback:
            tname, targs = fallback
            tool_calls = [{"name": tname, "args": targs, "id": f"call_{tname}"}]
            response.tool_calls = tool_calls
            response.content = ""

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

        # Sanitize argument types and fix schema artifacts
        clean_args = dict(tool_args)
        if tool_name == "generate_quiz":
            try:
                clean_args["number_of_questions"] = int(clean_args.get("number_of_questions", 5))
            except Exception:
                clean_args["number_of_questions"] = 5
            if str(clean_args.get("topic", "")).lower() in ("string", "", "none"):
                clean_args["topic"] = "DBMS joins"
        elif tool_name == "create_study_plan":
            try:
                clean_args["available_hours"] = float(clean_args.get("available_hours", 3.0))
            except Exception:
                clean_args["available_hours"] = 3.0
            if str(clean_args.get("subjects", "")).lower() in ("string", "", "none"):
                clean_args["subjects"] = "DSA and Operating Systems"
        elif tool_name == "calculate":
            if "expression" not in clean_args:
                clean_args["expression"] = clean_args.get("query", clean_args.get("math_expression", ""))
        elif tool_name in ("search_documents", "retrieve_memory"):
            if "query" not in clean_args:
                clean_args["query"] = clean_args.get("topic", clean_args.get("question", ""))
        elif tool_name == "summarize_document":
            if "doc_name" not in clean_args:
                clean_args["doc_name"] = clean_args.get("document_name", "")
        elif tool_name == "generate_quiz_from_document":
            if "doc_name" not in clean_args:
                clean_args["doc_name"] = clean_args.get("document_name", "")
            try:
                clean_args["num_questions"] = int(clean_args.get("num_questions", 5))
            except Exception:
                clean_args["num_questions"] = 5

        try:
            # Execute tool safely
            raw_output = tool.invoke(clean_args)
            output_str = str(raw_output)

            # If search_documents was executed, record in retrieved_documents for citations
            if tool_name == "search_documents":
                prior_retrieved_docs.append({"query": clean_args.get("query", ""), "result": output_str})

            tool_messages.append(ToolMessage(content=output_str, tool_call_id=tool_id, name=tool_name))
            current_results.append({"tool": tool_name, "args": clean_args, "result": output_str, "status": "success"})

        except Exception as e:
            # Graceful error recovery: inform the agent politely without raw tracebacks
            friendly_error = (
                f"Notice: The tool '{tool_name}' encountered an error while processing: {str(e)}. "
                "Please notify the student politely and answer to the best of your general ability."
            )
            tool_messages.append(ToolMessage(content=friendly_error, tool_call_id=tool_id, name=tool_name))
            current_results.append({"tool": tool_name, "args": clean_args, "result": friendly_error, "status": "error"})

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
    Guarantees raw JSON function calls are never exposed to the student.
    """
    content = ""
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage) and m.content and m.content.strip():
            raw_txt = m.content.strip()
            # If the content looks like a raw JSON function call, don't show it to the user!
            if ('"function"' in raw_txt or '"type":' in raw_txt) and ('"name"' in raw_txt or '"parameters"' in raw_txt):
                continue
            content = raw_txt
            break

    # If assistant content is empty or was raw tool JSON, fallback to the tool output
    if not content or content.strip() == "":
        tool_results = state.get("tool_results", [])
        if tool_results:
            content = tool_results[-1].get("result", "")

    # Safety fallback if content is still empty
    if not content or content.strip() == "":
        content = "I have processed your request. Please let me know if you need more details or further assistance!"

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
    Routes to 'tools' if the agent requested tool calls and tools have not yet been executed,
    otherwise routes to 'final_response'.
    """
    messages = state.get("messages", [])
    if not messages:
        return "final_response"

    # Prevent infinite loop: if tools have already run in the CURRENT turn, route directly to final_response
    last_human_idx = -1
    for idx, m in enumerate(messages):
        if isinstance(m, HumanMessage):
            last_human_idx = idx
    messages_in_turn = messages[last_human_idx:] if last_human_idx >= 0 else messages
    if any(isinstance(m, ToolMessage) for m in messages_in_turn):
        return "final_response"

    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", [])

    if tool_calls:
        return "tools"
    return "final_response"
