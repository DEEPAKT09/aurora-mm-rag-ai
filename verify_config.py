from multimodal_rag.app.config import PROJECT_ROOT, settings

print("Project root:", PROJECT_ROOT)
print("Upload directory:", settings.upload_dir)
print("Processed directory:", settings.processed_dir)
print("Index directory:", settings.index_dir)
print("Metadata directory:", settings.metadata_dir)

print("\nChecking paths:")

for name, path in {
    "Project root": PROJECT_ROOT,
    "Upload": settings.upload_dir,
    "Processed": settings.processed_dir,
    "Indexes": settings.index_dir,
    "Metadata": settings.metadata_dir,
}.items():

    print(f"{name}: {path}")
    print(f"  Exists: {path.exists()}")