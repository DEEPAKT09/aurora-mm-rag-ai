from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).parent
RAG_ROOT = PROJECT_ROOT / "multimodal_rag"

patterns = [
    r"^\s*from\s+app(?:\.|\s+import)",
    r"^\s*import\s+app(?:\.|\s|$)",
]

print("Searching for old top-level app imports...\n")

for file_path in RAG_ROOT.rglob("*.py"):

    source = file_path.read_text(encoding="utf-8")

    for line_number, line in enumerate(source.splitlines(), start=1):

        if any(re.search(pattern, line) for pattern in patterns):
            relative_path = file_path.relative_to(PROJECT_ROOT)

            print(f"{relative_path}:{line_number}")
            print(f"    {line.strip()}")