"""Security helpers for uploaded PDF documents."""

from pathlib import Path
from uuid import uuid4
import hashlib
from fastapi import HTTPException, UploadFile

from multimodal_rag.app.config import settings


PDF_SIGNATURE = b"%PDF-"


def validate_pdf_metadata(file: UploadFile) -> None:
    """Validate the supplied filename extension and declared MIME type."""

    filename = file.filename or ""

    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed.",
        )

    if file.content_type not in {
        "application/pdf",
        "application/octet-stream",
    }:
        raise HTTPException(
            status_code=400,
            detail="Invalid file content type. Upload a PDF.",
        )


def create_upload_path() -> Path:
    """Create a unique server-controlled destination path."""

    upload_dir = settings.upload_dir.resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_path = (upload_dir / f"{uuid4().hex}.pdf").resolve()

    # Defense in depth: the destination must remain inside upload_dir.
    if saved_path.parent != upload_dir:
        raise HTTPException(
            status_code=500,
            detail="Could not create a safe upload destination.",
        )

    return saved_path


def validate_pdf_signature(first_bytes: bytes) -> None:
    """Reject files that do not begin with the PDF signature."""

    if not first_bytes.startswith(PDF_SIGNATURE):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid PDF.",
        )


def get_max_upload_bytes() -> int:
    """Convert the configured upload limit from MiB to bytes."""

    return settings.max_upload_size_mb * 1024 * 1024


def validate_pdf_file(
    file_path: Path,
    max_bytes: int | None = None,
) -> tuple[int, str]:
    """Validate a PDF and return its size and SHA-256 hash."""

    file_path = Path(file_path)

    # 1. Validate the extension.
    if file_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are allowed.")

    # 2. Check that the file exists.
    if not file_path.is_file():
        raise ValueError("PDF file does not exist.")

    # 3. Determine the maximum permitted size.
    if max_bytes is None:
        max_bytes = settings.max_upload_size_mb * 1024 * 1024

    # 4. Check the file size.
    file_size = file_path.stat().st_size

    if file_size > max_bytes:
        raise ValueError(
            f"PDF exceeds the maximum allowed size of {max_bytes} bytes."
        )

    # 5. Validate the PDF signature and calculate SHA-256.
    hasher = hashlib.sha256()

    with file_path.open("rb") as file:
        signature = file.read(5)

        if signature != b"%PDF-":
            raise ValueError("The file does not have a valid PDF signature.")

        hasher.update(signature)

        for block in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(block)

    # 6. Return both values expected by the test.
    return file_size, hasher.hexdigest()



def resolve_within(base_dir: Path, candidate: Path) -> Path:
    """Resolve a path and reject paths outside the permitted directory."""

    base = Path(base_dir).resolve()
    resolved = Path(candidate).resolve()

    if not resolved.is_relative_to(base):
        raise ValueError("Path is outside the permitted directory.")

    return resolved

def safe_upload_path(
    root_dir: Path | None = None,
    filename: str | None = None,
) -> Path:
    """Generate a unique PDF path inside the specified directory."""

    # Use the test directory when supplied; otherwise use application storage.
    upload_dir = Path(root_dir or settings.upload_dir).resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Never use an untrusted filename to construct the destination.
    safe_name = f"{uuid4().hex}.pdf"
    candidate = (upload_dir / safe_name).resolve()

    return resolve_within(upload_dir, candidate)

