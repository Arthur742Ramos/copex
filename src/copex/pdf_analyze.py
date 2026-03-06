"""PDF rendering helpers for Copex vision-enabled tools."""

from __future__ import annotations

import base64
import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from copex.security import SecurityError, validate_path

_PDF_RENDER_DPI = 200


class PdfAnalyzeError(ValueError):
    """Raised when PDF analysis inputs cannot be prepared."""


@dataclass(frozen=True)
class RenderedPdfPage:
    """Rendered PDF page content packaged as a PNG image."""

    page_number: int
    png_bytes: bytes

    def to_binary_result(self, *, source_name: str) -> dict[str, str]:
        return {
            "data": base64.b64encode(self.png_bytes).decode("ascii"),
            "mimeType": "image/png",
            "type": "image",
            "description": f"Rendered PDF page {self.page_number} from {source_name}",
        }


@dataclass(frozen=True)
class PdfRenderBatch:
    """Rendered PDF pages plus metadata about the source document."""

    source_path: Path
    display_path: str
    page_count: int
    rendered_pages: tuple[RenderedPdfPage, ...]

    @property
    def selected_pages(self) -> tuple[int, ...]:
        return tuple(page.page_number for page in self.rendered_pages)

    @property
    def total_image_bytes(self) -> int:
        return sum(len(page.png_bytes) for page in self.rendered_pages)


@dataclass(frozen=True)
class PdfAnalysisPayload:
    """Prepared PDF page images and analysis instructions for the SDK tool."""

    source_path: Path
    display_path: str
    selected_pages: tuple[int, ...]
    prompt_text: str
    binary_results: list[dict[str, str]]
    total_image_bytes: int


def pdf_support_available() -> bool:
    """Return True when the PyMuPDF dependency is available."""

    try:
        importlib.import_module("fitz")
    except ImportError:
        return False
    return True


def parse_page_selection(page_spec: str | None, *, page_count: int) -> list[int]:
    """Parse user page selections like ``1-3,5`` into sorted page numbers."""

    if page_count < 1:
        raise PdfAnalyzeError("PDF has no pages to render.")

    if page_spec is None:
        return list(range(1, page_count + 1))

    normalized = page_spec.replace(" ", "").strip()
    if not normalized:
        raise PdfAnalyzeError("Page selection cannot be empty.")

    selected: set[int] = set()
    for part in normalized.split(","):
        if not part:
            raise PdfAnalyzeError(f"Invalid page selection: {page_spec}")
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = _parse_page_number(start_text, page_spec)
            end = _parse_page_number(end_text, page_spec)
            if start > end:
                raise PdfAnalyzeError(
                    f"Invalid page range '{part}': start page must be less than or equal to end page."
                )
            selected.update(range(start, end + 1))
            continue
        selected.add(_parse_page_number(part, page_spec))

    pages = sorted(selected)
    invalid = [page for page in pages if page > page_count]
    if invalid:
        raise PdfAnalyzeError(
            f"Requested page(s) {', '.join(str(page) for page in invalid)} exceed PDF page count "
            f"({page_count})."
        )
    return pages


def render_pdf_pages(
    path: str | Path,
    *,
    working_dir: Path,
    pages: str | None = None,
) -> PdfRenderBatch:
    """Render selected PDF pages to PNG bytes at 200 DPI."""

    fitz = _load_fitz()
    working_root = working_dir.resolve()
    pdf_path = _resolve_user_path(path, working_dir=working_root, must_exist=True)
    if not pdf_path.is_file():
        raise PdfAnalyzeError(f"PDF path must point to a file: {_display_path(pdf_path, working_root)}")

    open_errors = _fitz_error_types(fitz)
    try:
        document = fitz.open(str(pdf_path))
    except open_errors as exc:
        raise PdfAnalyzeError(f"Failed to open PDF '{_display_path(pdf_path, working_root)}': {exc}") from exc

    try:
        if not bool(getattr(document, "is_pdf", True)):
            raise PdfAnalyzeError(f"File is not a PDF: {_display_path(pdf_path, working_root)}")
        page_count = int(document.page_count)
        page_numbers = parse_page_selection(pages, page_count=page_count)
        scale = _PDF_RENDER_DPI / 72.0
        matrix = fitz.Matrix(scale, scale)
        rendered_pages = tuple(
            RenderedPdfPage(
                page_number=page_number,
                png_bytes=document.load_page(page_number - 1)
                .get_pixmap(matrix=matrix, alpha=False)
                .tobytes("png"),
            )
            for page_number in page_numbers
        )
    except open_errors as exc:
        raise PdfAnalyzeError(
            f"Failed to render PDF pages from '{_display_path(pdf_path, working_root)}': {exc}"
        ) from exc
    finally:
        document.close()

    return PdfRenderBatch(
        source_path=pdf_path,
        display_path=_display_path(pdf_path, working_root),
        page_count=page_count,
        rendered_pages=rendered_pages,
    )


def prepare_pdf_analysis_payload(
    path: str | Path,
    *,
    prompt: str,
    working_dir: Path,
    pages: str | None = None,
) -> PdfAnalysisPayload:
    """Prepare rendered PDF pages and prompt text for an SDK tool result."""

    analysis_prompt = prompt.strip()
    if not analysis_prompt:
        raise PdfAnalyzeError("Analysis prompt cannot be empty.")

    batch = render_pdf_pages(path, working_dir=working_dir, pages=pages)
    page_ranges = _format_page_ranges(batch.selected_pages)
    prompt_text = "\n".join(
        [
            f"The user wants an analysis of PDF '{batch.display_path}'.",
            f"Rendered page images are attached for page(s): {page_ranges}.",
            f"User request: {analysis_prompt}",
            (
                "Inspect all visible text and visual elements, including charts, figures, tables, "
                "diagrams, and annotations. Answer directly, citing page numbers when useful and "
                "calling out uncertainty when details are unclear."
            ),
        ]
    )
    binary_results = [
        rendered_page.to_binary_result(source_name=batch.source_path.name)
        for rendered_page in batch.rendered_pages
    ]
    return PdfAnalysisPayload(
        source_path=batch.source_path,
        display_path=batch.display_path,
        selected_pages=batch.selected_pages,
        prompt_text=prompt_text,
        binary_results=binary_results,
        total_image_bytes=batch.total_image_bytes,
    )


def export_pdf_screenshots(
    path: str | Path,
    *,
    pages: str,
    output_dir: str | Path,
    working_dir: Path,
) -> list[Path]:
    """Render selected PDF pages to PNG files inside ``output_dir``."""

    page_spec = pages.strip()
    if not page_spec:
        raise PdfAnalyzeError("Page selection cannot be empty.")

    batch = render_pdf_pages(path, working_dir=working_dir, pages=page_spec)
    working_root = working_dir.resolve()
    destination_dir = _resolve_user_path(output_dir, working_dir=working_root, must_exist=False)
    if destination_dir.exists() and not destination_dir.is_dir():
        raise PdfAnalyzeError(
            f"Output directory must be a directory: {_display_path(destination_dir, working_root)}"
        )
    destination_dir.mkdir(parents=True, exist_ok=True)

    width = max(3, len(str(batch.page_count)))
    written_paths: list[Path] = []
    for rendered_page in batch.rendered_pages:
        filename = f"{batch.source_path.stem}-page-{rendered_page.page_number:0{width}d}.png"
        output_path = destination_dir / filename
        output_path.write_bytes(rendered_page.png_bytes)
        written_paths.append(output_path)
    return written_paths


def _parse_page_number(value: str, page_spec: str) -> int:
    try:
        page = int(value)
    except (TypeError, ValueError) as exc:
        raise PdfAnalyzeError(f"Invalid page selection: {page_spec}") from exc
    if page < 1:
        raise PdfAnalyzeError(f"Page numbers must be 1 or greater: {page_spec}")
    return page


def _load_fitz() -> Any:
    try:
        return importlib.import_module("fitz")
    except ImportError as exc:
        raise PdfAnalyzeError(
            "PyMuPDF is not installed. Install the 'pdf' extra or the 'pymupdf' package "
            "to use PDF tools."
        ) from exc


def _fitz_error_types(fitz: Any) -> tuple[type[BaseException], ...]:
    error_types: list[type[BaseException]] = [OSError, RuntimeError, ValueError, TypeError]
    for name in ("FileDataError", "EmptyFileError"):
        error_type = getattr(fitz, name, None)
        if isinstance(error_type, type) and issubclass(error_type, BaseException):
            error_types.append(error_type)
    return tuple(dict.fromkeys(error_types))


def _resolve_user_path(path: str | Path, *, working_dir: Path, must_exist: bool) -> Path:
    raw_path = Path(path)
    candidate = raw_path if raw_path.is_absolute() else working_dir / raw_path
    try:
        return validate_path(candidate, base_dir=working_dir, must_exist=must_exist, allow_absolute=True)
    except SecurityError as exc:
        raise PdfAnalyzeError(str(exc)) from exc


def _display_path(path: Path, working_dir: Path) -> str:
    try:
        return str(path.relative_to(working_dir))
    except ValueError:
        return str(path)


def _format_page_ranges(page_numbers: Sequence[int]) -> str:
    if not page_numbers:
        return ""

    ranges: list[str] = []
    start = end = page_numbers[0]
    for page in page_numbers[1:]:
        if page == end + 1:
            end = page
            continue
        ranges.append(f"{start}-{end}" if start != end else str(start))
        start = end = page
    ranges.append(f"{start}-{end}" if start != end else str(start))
    return ",".join(ranges)
