import importlib
import sys

packages = [
    "streamlit",
    "langchain_google_genai",
    "google.genai",
    "langchain_core",
    "langchain_text_splitters",
    "langchain_chroma",
    "chromadb",
    "langgraph",
    "fitz",
    "pydantic",
    "pydantic_settings",
    "dotenv",
    "requests",
]

print("Python executable:", sys.executable)
print("\nPackage verification")
print("=" * 50)

for package in packages:
    try:
        importlib.import_module(package)
        print(f"[OK] {package}")
    except Exception as error:
        print(f"[FAILED] {package}: {error}")