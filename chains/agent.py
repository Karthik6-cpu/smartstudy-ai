"""
Autonomous Study Agent with LangChain Tool Calling.
Enables local Ollama LLMs to intelligently decide when to invoke:
- search_documents
- calculate
- generate_quiz
- create_study_plan
- save_memory / retrieve_memory
"""

import json
import re
from typing import List, Dict, Any, Tuple, Generator, Optional
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
)

from llm.langchain_client import get_chat_ollama, LangChainOllamaFactory
from tools.study_tools import ALL_STUDY_TOOLS, TOOLS_BY_NAME
from config.settings import AGENT_SYSTEM_PROMPT, DEFAULT_MODEL, DEFAULT_OLLAMA_HOST


class StudyAgent:
    """Orchestrates autonomous tool calling and final response generation."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_OLLAMA_HOST,
        temperature: float = 0.2,
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature

    def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Safely execute a registered tool by name with arguments."""
        tool = TOOLS_BY_NAME.get(tool_name)
        if not tool:
            return f"Error: Tool '{tool_name}' not found."
        try:
            result = tool.invoke(tool_args)
            return str(result)
        except Exception as e:
            return f"Error executing tool '{tool_name}': {str(e)}"

    def detect_intent_fallback(self, query: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Heuristic fallback router if local LLM does not generate structured tool_calls.
        Ensures guaranteed execution for explicit user commands.
        """
        q_lower = query.lower().strip()

        # 1. Save memory pattern (e.g. 'Remember that my target exam score is 90%')
        if q_lower.startswith("remember that") or q_lower.startswith("remember:") or "remember that" in q_lower:
            fact = re.sub(r"^.*?remember\s*(that|:)?\s*", "", query, flags=re.IGNORECASE).strip(" .?!,;")
            if " is " in fact:
                parts = fact.split(" is ", 1)
                return "save_memory", {"key": parts[0].strip(), "value": parts[1].strip()}
            elif "=" in fact:
                parts = fact.split("=", 1)
                return "save_memory", {"key": parts[0].strip(), "value": parts[1].strip()}
            else:
                return "save_memory", {"key": "target_score", "value": fact}

        # 2. Retrieve memory pattern (e.g. 'What is my target exam score?', 'What do you remember about me?')
        if any(p in q_lower for p in ["what do you remember", "recall my", "what is my", "what's my", "target exam score", "target score"]):
            search_key = re.sub(r"(what\s+(is|do\s+you\s+remember|are)\s+(my)?|recall\s+my)\s*", "", query, flags=re.IGNORECASE).strip(" ?.,!:")
            return "retrieve_memory", {"query": search_key or "target exam score"}

        # 3. PDF Summarization pattern (e.g. 'Summarize this PDF', 'Summarize my notes', 'Summarize it', 'I uploaded my notes. Summarize them.')
        if any(phrase in q_lower for phrase in ["summarize this pdf", "summarize the pdf", "summarize my notes", "summarize these notes", "summarize this document", "summarize it", "summarize them", "give me a summary of my notes", "give me a summary of this pdf", "pdf summary"]):
            return "summarize_document", {"doc_name": ""}

        # 4. PDF Quiz / Exam Questions pattern (e.g. 'Give me 10 questions from this PDF', 'Give me 5 MCQs from these notes', 'exam questions from this')
        if any(phrase in q_lower for phrase in ["from this pdf", "from the pdf", "from these notes", "from my notes", "from this document", "from this"]) and any(w in q_lower for w in ["questions", "mcqs", "mcq", "quiz", "test"]):
            num = 5
            num_match = re.search(r"(\d+)\s*(?:questions|mcqs|mcq)", q_lower)
            if num_match:
                num = int(num_match.group(1))
            return "generate_quiz_from_document", {"doc_name": "", "num_questions": num}

        # 5. Uploaded notes / PDF search pattern (e.g. 'from my uploaded notes', 'what does the PDF say about CMMI?', 'according to these notes', 'in these notes')
        if any(phrase in q_lower for phrase in ["from my uploaded", "in my notes", "from my notes", "uploaded notes", "my uploaded", "doc search", "what does the pdf say", "in the pdf", "from the pdf", "according to these notes", "in these notes", "after cmmi in these notes"]):
            clean_topic = re.sub(r"(from|in|according\s+to)\s+(my|these|the)?\s*(uploaded\s+)?(notes|documents|syllabus|files|pdf)", "", query, flags=re.IGNORECASE).strip(" .?!,;")
            clean_topic = re.sub(r"^(what\s+does\s+the\s+pdf\s+say\s+about|explain|describe|what is|summarize)\s*", "", clean_topic, flags=re.IGNORECASE).strip(" .?!,;")
            return "search_documents", {"query": clean_topic or "CMMI"}

        # 6. Quiz request pattern (e.g. 'Give me 5 MCQs about DBMS joins', 'Quiz on DSA')
        if any(word in q_lower for word in ["mcqs", "mcq", "quiz", "practice questions", "test questions"]):
            num = 5
            num_match = re.search(r"(\d+)\s*(?:mcqs|questions|quiz)", q_lower)
            if num_match:
                num = int(num_match.group(1))
            topic_match = re.search(r"(?:about|on|for|in)\s+(.+)", query, flags=re.IGNORECASE)
            raw_topic = topic_match.group(1) if topic_match else query
            clean_topic = re.sub(r"(?:give me|generate|create|prepare|\d+\s*mcqs|\d+\s*questions)\s*", "", raw_topic, flags=re.IGNORECASE).strip(" .?!,;")
            return "generate_quiz", {"topic": clean_topic or "DBMS joins", "number_of_questions": num}

        # 7. Study plan pattern (e.g. 'Create a 30-day study plan for TCS NQT', 'make me a study plan')
        if any(p in q_lower for p in ["create a study plan", "study plan for", "revision timetable", "revision schedule", "study plan", "make me a study plan", "exam preparation schedule"]):
            day_match = re.search(r"(\d+)[-\s]*day", q_lower)
            timeline = f"in {day_match.group(1)} days" if day_match else "in 4 weeks"
            sub_match = re.search(r"(?:for|on)\s+(.+?)(?:\s+for\s+the\s+next|\s+in\s+|\.|$)", query, flags=re.IGNORECASE)
            subjects = sub_match.group(1).strip() if sub_match else "MCA Core Subjects"
            return "create_study_plan", {"subjects": subjects, "available_hours": 3.0, "exam_date": timeline}

        # 8. MCA syllabus / curriculum overview pattern
        if any(p in q_lower for p in ["syllabus", "curriculum", "subject overview", "topics in", "units in", "course overview"]):
            for sub in ["dsa", "data structures", "dbms", "database", "operating systems", "os", "computer networks", "networks", "software engineering", "se"]:
                if sub in q_lower:
                    return "get_mca_subject_overview", {"subject_name": sub}
            return "get_mca_subject_overview", {"subject_name": query.strip()}

        # 9. Math calculation pattern (e.g. 'What is 20 added to 30?', 'Calculate 25% of 480', 'two plus two', '50 * 12', 'sqrt(144)')
        added_match = re.search(r"(\d+(?:\.\d+)?|[a-z]+)\s+added\s+to\s+(\d+(?:\.\d+)?|[a-z]+)", q_lower)
        if added_match:
            expr = f"{added_match.group(1)} + {added_match.group(2)}"
            return "calculate", {"expression": expr}

        math_operators = ["+", "-", "*", "/", "%", "^", "plus", "minus", "times", "multiplied by", "divided by", "square root", "sqrt", "% of", "percent of", "to the power of"]
        has_math_op = any(re.search(rf"\b{re.escape(op)}\b" if op.replace(" ", "").isalpha() else re.escape(op), q_lower) for op in math_operators)

        num_words_list = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million"]
        has_digits_or_num_words = any(char.isdigit() for char in q_lower) or any(re.search(rf"\b{nw}\b", q_lower) for nw in num_words_list)

        calc_match = re.search(r"(?:calculate|compute|what\s+is|solve|evaluate|how\s+much\s+is)\s+([0-9a-z\.\s\+\-\*\/\%\^\(\)]+(?:of\s+[0-9a-z\.]+)?|sqrt\([^)]+\))", q_lower)
        if calc_match and (has_math_op or any(char.isdigit() for char in q_lower)):
            expr = calc_match.group(1).strip().rstrip("?.!;, ")
            return "calculate", {"expression": expr}
        elif has_math_op and has_digits_or_num_words:
            clean_expr = re.sub(r"^(?:calculate|what\s+is|compute|solve|evaluate|how\s+much\s+is|can\s+you\s+calculate|find)\s*", "", query, flags=re.IGNORECASE).strip().rstrip("?.!;, ")
            return "calculate", {"expression": clean_expr}

        # 10. Study schedule calculation
        if any(p in q_lower for p in ["study schedule", "revision calculation", "hours per subject", "calculate study", "schedule calculation"]):
            return "calculate_study_schedule", {"subjects": "DSA, DBMS, OS, Networks", "total_weeks": 4, "daily_hours": 3.0}

        return None

    def run(
        self,
        messages: List[Dict[str, str]],
    ) -> Tuple[Generator[str, None, None], List[Dict[str, Any]]]:
        """
        Run agent turn with dynamic tool calling.

        Returns:
            Tuple of (token_stream_generator, executed_tools_list)
        """
        # Connection check
        is_ok, conn_msg = LangChainOllamaFactory.validate_connection(self.base_url)
        if not is_ok:
            def err_gen():
                yield f"⚠️ **Ollama is offline:**\n\n{conn_msg}"
            return err_gen(), []

        # Convert dict messages to LangChain BaseMessage instances
        lc_messages: List[BaseMessage] = [SystemMessage(content=AGENT_SYSTEM_PROMPT)]
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))

        executed_tools = []
        latest_user_query = messages[-1].get("content", "") if messages else ""

        try:
            llm = get_chat_ollama(
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature,
            )

            # Bind tools to model
            llm_with_tools = llm.bind_tools(ALL_STUDY_TOOLS)

            # 1. Initial invocation to check if LLM decides to call tools
            initial_ai_msg = llm_with_tools.invoke(lc_messages)
            tool_calls = getattr(initial_ai_msg, "tool_calls", [])

            # If model didn't return structured tool_calls, check intent heuristic
            if not tool_calls:
                fallback_intent = self.detect_intent_fallback(latest_user_query)
                if fallback_intent:
                    t_name, t_args = fallback_intent
                    tool_calls = [{"name": t_name, "args": t_args, "id": f"call_{t_name}"}]

            # If tools were invoked
            if tool_calls:
                lc_messages.append(initial_ai_msg)

                for tcall in tool_calls:
                    tname = tcall.get("name")
                    targs = tcall.get("args", {})
                    tid = tcall.get("id", f"call_{tname}")

                    tool_result = self.execute_tool(tname, targs)
                    executed_tools.append({
                        "tool": tname,
                        "args": targs,
                        "result": tool_result,
                    })

                    tool_msg = ToolMessage(
                        content=tool_result,
                        tool_call_id=tid,
                        name=tname,
                    )
                    lc_messages.append(tool_msg)

                # Final synthesis stream
                def stream_after_tools():
                    for chunk in llm.stream(lc_messages):
                        yield chunk.content

                return stream_after_tools(), executed_tools

            else:
                # No tool called: direct answer stream
                def stream_direct():
                    if hasattr(initial_ai_msg, "content") and initial_ai_msg.content:
                        yield initial_ai_msg.content
                    else:
                        for chunk in llm.stream(lc_messages):
                            yield chunk.content

                return stream_direct(), []

        except Exception as e:
            # Fallback error generator
            def exc_gen():
                yield f"\n\n❌ **Agent Execution Error:** `{str(e)}`"
            return exc_gen(), executed_tools
