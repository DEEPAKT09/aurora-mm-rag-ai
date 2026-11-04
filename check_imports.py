import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

SKIP_DIRS = {
    ".venv",
    "__pycache__",
    ".git",
}

imports = {}

for python_file in PROJECT_ROOT.rglob("*.py"):

    if any(part in SKIP_DIRS for part in python_file.parts):
        continue

    try:
        source = python_file.read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):

            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    imports.setdefault(module, set()).add(
                        str(python_file.relative_to(PROJECT_ROOT))
                    )

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split(".")[0]
                    imports.setdefault(module, set()).add(
                        str(python_file.relative_to(PROJECT_ROOT))
                    )

    except (SyntaxError, UnicodeDecodeError) as error:
        print(f"Could not inspect {python_file}: {error}")

print("\nEXTERNAL IMPORT INVENTORY")
print("=" * 60)

for module in sorted(imports):
    print(f"\n{module}")
    for source_file in sorted(imports[module]):
        print(f"  - {source_file}")