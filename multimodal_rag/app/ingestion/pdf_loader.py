from pathlib import Path
import hashlib
import logging
import re
import shutil

import pymupdf

from multimodal_rag.app.config import settings, PROJECT_ROOT
from multimodal_rag.app.schemas import DocumentRecord, ContentChunk


logger = logging.getLogger(__name__)


def calculate_file_hash(file_path: Path) -> str:
    """
    Calculate the SHA-256 hash of a PDF.

    The file is read in small blocks rather than loaded entirely
    into memory.
    """

    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(
            f"PDF file not found: {file_path}"
        )

    hasher = hashlib.sha256()

    with file_path.open("rb") as file:

        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            hasher.update(block)

    return hasher.hexdigest()


def extract_pdf(
    pdf_path: str | Path,
    document_id_override: str | None = None,
) -> dict:
    """
    Extract text, tables, embedded images, and rendered page images.

    document_id_override:
        Optional ID supplied by IndexManager.

        When omitted, the original content-hash-based document ID
        is retained for standalone MM-RAG compatibility.
    """

    pdf_path = Path(pdf_path).resolve()

    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            "The supplied file must have a .pdf extension."
        )

    # ---------------------------------------------------------
    # 1. Calculate the PDF content hash.
    # ---------------------------------------------------------

    content_hash = calculate_file_hash(pdf_path)

    # Preserve the original ID format when no override is supplied.
    if document_id_override is None:

        document_id = f"doc_{content_hash[:16]}"

    else:

        if not re.fullmatch(
            r"doc_[a-f0-9]{16,64}",
            document_id_override,
        ):
            raise ValueError(
                "Invalid document ID format."
            )

        document_id = document_id_override

    document = DocumentRecord(
        document_id=document_id,
        source_name=pdf_path.name,
        file_path=str(pdf_path),
        content_hash=content_hash,
    )

    # ---------------------------------------------------------
    # 2. Prepare a safe asset directory.
    # ---------------------------------------------------------

    asset_dir = (
        settings.processed_dir / document_id
    ).resolve()

    processed_root = settings.processed_dir.resolve()

    if not asset_dir.is_relative_to(processed_root):
        raise ValueError(
            "Unsafe processed asset directory."
        )

    asset_dir_existed = asset_dir.exists()

    asset_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunks: list[ContentChunk] = []

    warnings: list[str] = []

    embedded_images_extracted = 0

    try:

        with pymupdf.open(pdf_path) as pdf:

            # -------------------------------------------------
            # 3. Validate the PDF.
            # -------------------------------------------------

            if pdf.needs_pass:
                raise ValueError(
                    "This PDF is password-protected and cannot "
                    "be opened without authentication."
                )

            if pdf.page_count == 0:
                raise ValueError(
                    "The PDF contains no pages."
                )

            if pdf.page_count > settings.max_pdf_pages:
                raise ValueError(
                    f"This PDF contains {pdf.page_count} pages. "
                    f"The current maximum is "
                    f"{settings.max_pdf_pages} pages."
                )

            # -------------------------------------------------
            # 4. Process one page at a time.
            # -------------------------------------------------

            for page_number, page in enumerate(
                pdf,
                start=1,
            ):

                # ---------------------------------------------
                # A. Extract selectable text.
                # ---------------------------------------------

                text = page.get_text("text").strip()

                # ---------------------------------------------
                # B. Attempt OCR if selectable text is absent.
                # ---------------------------------------------

                if not text:

                    try:

                        text_page = page.get_textpage_ocr()

                        text = page.get_text(
                            "text",
                            textpage=text_page,
                        ).strip()

                    except Exception:

                        warning = (
                            f"Page {page_number}: OCR was "
                            "unavailable or failed."
                        )

                        warnings.append(warning)

                        logger.warning(
                            "OCR failed for document_id=%s page=%s",
                            document_id,
                            page_number,
                            exc_info=True,
                        )

                if text:

                    chunks.append(
                        ContentChunk(
                            chunk_id=(
                                f"{document_id}_p{page_number}_text_001"
                            ),
                            document_id=document_id,
                            page_number=page_number,
                            content_type="text",
                            text=text,
                        )
                    )

                # ---------------------------------------------
                # C. Extract tables.
                # ---------------------------------------------

                try:

                    table_finder = page.find_tables()

                    for table_number, table in enumerate(
                        table_finder.tables,
                        start=1,
                    ):

                        table_text = table.to_markdown().strip()

                        if table_text:

                            chunks.append(
                                ContentChunk(
                                    chunk_id=(
                                        f"{document_id}_p{page_number}"
                                        f"_table_{table_number:03d}"
                                    ),
                                    document_id=document_id,
                                    page_number=page_number,
                                    content_type="table",
                                    text=table_text,
                                )
                            )

                except Exception as exc:

                    warning = (
                        f"Page {page_number}: table extraction failed "
                        f"({exc})."
                    )

                    warnings.append(warning)

                    logger.warning(warning)

                # ---------------------------------------------
                # D. Extract embedded images with a document cap.
                # ---------------------------------------------

                if (
                    embedded_images_extracted
                    < settings.max_embedded_images_per_pdf
                ):

                    for image_number, image_info in enumerate(
                        page.get_images(full=True),
                        start=1,
                    ):

                        if (
                            embedded_images_extracted
                            >= settings.max_embedded_images_per_pdf
                        ):
                            warnings.append(
                                "The embedded-image extraction limit "
                                "was reached. Remaining embedded images "
                                "were skipped."
                            )
                            break

                        xref = image_info[0]

                        try:

                            image_data = pdf.extract_image(xref)

                            image_bytes = image_data["image"]

                            image_extension = image_data["ext"]

                            image_path = (
                                asset_dir
                                / f"page_{page_number:04d}"
                                f"_image_{image_number:03d}."
                                f"{image_extension}"
                            )

                            image_path.write_bytes(image_bytes)

                            chunks.append(
                                ContentChunk(
                                    chunk_id=(
                                        f"{document_id}_p{page_number}"
                                        f"_image_{image_number:03d}"
                                    ),
                                    document_id=document_id,
                                    page_number=page_number,
                                    content_type="image",
                                    asset_path=str(image_path),
                                )
                            )

                            embedded_images_extracted += 1

                        except Exception as exc:

                            warning = (
                                f"Page {page_number}, image "
                                f"{image_number}: extraction failed "
                                f"({exc})."
                            )

                            warnings.append(warning)

                            logger.warning(warning)

                # ---------------------------------------------
                # E. Render the complete page.
                # ---------------------------------------------

                page_image_path = (
                    asset_dir / f"page_{page_number:04d}.png"
                )

                try:

                    page.get_pixmap(
                        dpi=settings.page_render_dpi,
                        alpha=False,
                    ).save(page_image_path)

                    chunks.append(
                        ContentChunk(
                            chunk_id=(
                                f"{document_id}_p{page_number}_page_image"
                            ),
                            document_id=document_id,
                            page_number=page_number,
                            content_type="page_image",
                            asset_path=str(page_image_path),
                        )
                    )

                except Exception as exc:

                    warning = (
                        f"Page {page_number}: page rendering failed "
                        f"({exc})."
                    )

                    warnings.append(warning)

                    logger.warning(warning)

    except Exception:

        logger.exception(
            "Failed to ingest PDF: %s",
            pdf_path,
        )

        # Remove assets only if this attempt created the directory.
        # Never remove an existing directory that may be in use.
        if not asset_dir_existed:

            shutil.rmtree(
                asset_dir,
                ignore_errors=True,
            )

        raise

    return {
        "document": document,
        "chunks": chunks,
        "warnings": warnings,
    }


if __name__ == "__main__":

    sample_pdf = (
        PROJECT_ROOT
        / "data"
        / "uploads"
        / "sample.pdf"
    )

    result = extract_pdf(sample_pdf)

    print("\nDocument:")

    print(
        result["document"].model_dump()
    )

    print("\nExtracted records:")

    for chunk in result["chunks"]:

        print(
            f"ID: {chunk.chunk_id} | "
            f"Page: {chunk.page_number} | "
            f"Type: {chunk.content_type} | "
            f"Text length: {len(chunk.text)} | "
            f"Asset: {chunk.asset_path}"
        )

    print(
        f"\nTotal records: {len(result['chunks'])}"
    )

    if result["warnings"]:

        print("\nWarnings:")

        for warning in result["warnings"]:
            print("-", warning)