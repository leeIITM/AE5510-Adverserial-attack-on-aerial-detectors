#!/usr/bin/env python3
"""
Paper PDF Compilation Script
============================
Compiles the IEEE conference technical paper into a clean PDF using ReportLab
with all embedded figures and professional two-column typography.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Run compilation
compile_script = REPO_ROOT / "paper" / "compile_paper_pdf.py"

def main():
    print("Compiling IEEE Conference Paper PDF...")
    os.system(f'python "{compile_script}"')
    pdf_path = REPO_ROOT / "paper" / "AE5510_Problem1_Paper.pdf"
    if pdf_path.exists():
        print(f"SUCCESS: Paper compiled at {pdf_path} ({pdf_path.stat().st_size / (1024*1024):.2f} MB)")
    else:
        print("Error compiling paper.")

if __name__ == "__main__":
    main()
