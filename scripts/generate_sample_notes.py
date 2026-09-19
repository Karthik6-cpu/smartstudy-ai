"""
Utility script to generate sample MCA notes (CMMI & Software Engineering)
for testing the local RAG pipeline immediately.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_rag_pipeline import create_minimal_text_pdf
from config.settings import DOCUMENTS_DIR


def main():
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    sample_path = DOCUMENTS_DIR / "MCA_Software_Engineering_CMMI_Notes.pdf"

    pages = [
        # Page 1
        "MCA Department - Master of Computer Applications\n"
        "Course: Advanced Software Engineering (Unit 4: Quality & Process Models)\n\n"
        "Topic: Capability Maturity Model Integration (CMMI)\n\n"
        "Overview:\n"
        "CMMI is a process level improvement training and appraisal program developed by Carnegie Mellon University.\n"
        "It provides organizations with the essential elements of effective processes that ultimately improve their performance.\n"
        "CMMI models provide guidance for developing or improving processes that meet the business goals of an organization.",

        # Page 2
        "CMMI Maturity Levels:\n\n"
        "Level 1 - Initial:\n"
        "Processes are unpredictable, poorly controlled, and reactive. Work gets completed, but it frequently exceeds budget and schedule.\n\n"
        "Level 2 - Managed:\n"
        "Processes are planned, performed, measured, and controlled at the project level. Requirements are managed, and processes are verified against project plans.\n\n"
        "Level 3 - Defined:\n"
        "Processes are well characterized and understood, and are described in standards, procedures, tools, and methods. The organization's set of standard processes is established.\n\n"
        "Level 4 - Quantitatively Managed:\n"
        "The organization and projects establish quantitative objectives for quality and process performance and use them as criteria in managing processes.\n\n"
        "Level 5 - Optimizing:\n"
        "Continual process improvement based on a quantitative understanding of business objectives and performance needs. Focus is on innovation and technological transformation.",

        # Page 3
        "Differences between CMMI Staged vs Continuous Representations:\n\n"
        "1. Staged Representation:\n"
        "- Uses predefined sets of process areas to define an improvement path characterized by 5 maturity levels.\n"
        "- Provides a proven sequence of improvements, each serving as a foundation for the next.\n\n"
        "2. Continuous Representation:\n"
        "- Enables an organization to select a specific process area and improve processes related to it.\n"
        "- Uses 4 capability levels (Incomplete, Performed, Managed, Defined) to measure individual process improvements."
    ]

    pdf_bytes = create_minimal_text_pdf(pages)
    with open(sample_path, "wb") as f:
        f.write(pdf_bytes)

    print(f"✅ Generated sample MCA lecture notes at: {sample_path}")
    print(f"File size: {len(pdf_bytes)} bytes across {len(pages)} pages.")


if __name__ == "__main__":
    main()
