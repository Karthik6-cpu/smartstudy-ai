"""
Reusable Prompt Templates for SmartStudy AI using LangChain.
Provides structured prompt engineering for general chat, RAG, quizzes, and planning.
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)

# 1. General Study Chat Prompt Template
GENERAL_STUDY_SYSTEM_MESSAGE = """You are SmartStudy AI, an expert, patient, and pedagogical study assistant designed specifically for Master of Computer Applications (MCA) students.

Your objectives:
1. Explain core Computer Science concepts (Data Structures, Algorithms, DBMS, Operating Systems, Computer Networks, Software Engineering, OOP) with technical accuracy and intuitive analogies.
2. Structure your explanations using clear markdown headings, bullet points, and summary takeaways for quick exam preparation.
3. Provide working code snippets (Python, C++, Java, or SQL) and pseudocode with time and space complexities where appropriate.
4. Maintain an encouraging and academic tone.
"""

GENERAL_STUDY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", GENERAL_STUDY_SYSTEM_MESSAGE),
    MessagesPlaceholder("history", optional=True),
    ("human", "{question}"),
])


# 2. RAG Study Prompt Template
RAG_SYSTEM_MESSAGE = """You are SmartStudy AI, an intelligent local study assistant answering questions based on the MCA student's uploaded lecture notes and course materials.

Strict Grounding Guidelines:
1. Ground your answer in the provided "Context from Uploaded Notes" below.
2. Explicitly state in your response that the answer is derived from the uploaded documents.
3. Reference the source document name(s) and page number(s) where the facts were found.
4. If the context does not contain enough information to answer the question, state clearly and politely:
   "I could not find information about this topic in your uploaded notes."
   Do NOT invent, fabricate, or assume information outside of the provided context.
"""

RAG_HUMAN_TEMPLATE = """Context from Uploaded Notes:
==================================================
{context}
==================================================

Student Question:
{question}

Please answer the student's question adhering strictly to the grounding guidelines."""

RAG_STUDY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_MESSAGE),
    ("human", RAG_HUMAN_TEMPLATE),
])


# 3. Quiz Generation Prompt Template
QUIZ_SYSTEM_MESSAGE = """You are an MCA Computer Science examiner creating high-yield practice assessment questions to test conceptual understanding and exam readiness."""

QUIZ_HUMAN_TEMPLATE = """Generate a comprehensive practice quiz for MCA students on the following topic:

Topic / Context:
{topic}

Difficulty Level: {difficulty}
Number of Questions: {num_questions}

Format your output strictly in Markdown with the following sections:
1. **Multiple Choice Questions (MCQs):** Each question with 4 options (A, B, C, D).
2. **Short Answer / Conceptual Questions:** Testing core definitions, differences, or mechanisms.
3. **Coding / Problem-Solving Challenge:** A pseudocode or implementation task.
4. **Answer Key & Detailed Explanations:** Provide full explanations and the correct answer for every question at the end."""

QUIZ_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", QUIZ_SYSTEM_MESSAGE),
    ("human", QUIZ_HUMAN_TEMPLATE),
])


# 4. Study Planning Prompt Template
PLANNER_SYSTEM_MESSAGE = """You are an academic mentor and timetable specialist for Master of Computer Applications (MCA) students. You design realistic, high-retention study schedules using spaced repetition and active recall."""

PLANNER_HUMAN_TEMPLATE = """Design a structured semester study and revision timetable based on the following student requirements:

Subjects / Syllabus Topics:
{subjects}

Available Duration: {timeline}
Daily Available Study Hours: {daily_hours} hours/day
Student Goal: {goal}

Format the output in clear Markdown containing:
1. **Executive Strategy & Spaced Repetition Milestones:** Key checkpoints before the target exam date.
2. **Phase-by-Phase Weekly Breakdown:** Allocation of subjects, high-weightage topics, and revision buffers.
3. **Sample Daily Routine:** Optimized time blocks (pomodoro technique, theory vs practice coding).
4. **Mock Assessment & Self-Test Checkpoints:** Recommended dates for taking practice quizzes.
5. **Exam Day Preparation Tips:** Final revision checklist."""

STUDY_PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", PLANNER_SYSTEM_MESSAGE),
    ("human", PLANNER_HUMAN_TEMPLATE),
])
