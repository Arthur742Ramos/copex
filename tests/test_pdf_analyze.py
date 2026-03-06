from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

import copex.cli as cli
from copex.config import CopexConfig
from copex.pdf_analyze import (
    PdfAnalysisPayload,
    PdfAnalyzeError,
    export_pdf_screenshots,
    parse_page_selection,
    prepare_pdf_analysis_payload,
)
from copex.sdk_tools import (
    _build_pdf_analyze_tool,
    _build_pdf_screenshot_tool,
    build_domain_tools,
    list_domain_tools,
    register_pdf_tools,
)


def run(coro):
    return asyncio.run(coro)


def _invocation(args: dict) -> dict:
    return {
        "session_id": "session-1",
        "tool_call_id": "call-1",
        "tool_name": "tool",
        "arguments": args,
    }


def _sample_payload(path: Path) -> PdfAnalysisPayload:
    return PdfAnalysisPayload(
        source_path=path,
        display_path=path.name,
        selected_pages=(1, 2),
        prompt_text="Inspect the attached pages and summarize the visuals.",
        binary_results=[
            {
                "data": base64.b64encode(b"page-1").decode("ascii"),
                "mimeType": "image/png",
                "type": "image",
                "description": "Rendered PDF page 1 from report.pdf",
            },
            {
                "data": base64.b64encode(b"page-2").decode("ascii"),
                "mimeType": "image/png",
                "type": "image",
                "description": "Rendered PDF page 2 from report.pdf",
            },
        ],
        total_image_bytes=12,
    )


def _create_tiny_pdf(path: Path) -> None:
    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    try:
        page_one = document.new_page(width=320, height=240)
        page_one.insert_text((36, 48), "Quarterly revenue grew 12% year over year.")
        chart_box = fitz.Rect(36, 90, 260, 180)
        page_one.draw_rect(chart_box, color=(0, 0, 0), fill=(0.85, 0.92, 1.0))
        page_one.insert_text((48, 120), "Chart placeholder")

        page_two = document.new_page(width=320, height=240)
        page_two.insert_text((36, 48), "Figure 2: Market share trends.")
        page_two.draw_line((36, 150), (260, 90), color=(0.1, 0.3, 0.8), width=2)
        page_two.draw_line((36, 150), (260, 170), color=(0.8, 0.2, 0.2), width=2)

        document.save(path)
    finally:
        document.close()


def test_parse_page_selection_defaults_to_all_pages() -> None:
    assert parse_page_selection(None, page_count=4) == [1, 2, 3, 4]


def test_parse_page_selection_supports_ranges_and_dedupes() -> None:
    assert parse_page_selection("3,1-2,2,5-5", page_count=5) == [1, 2, 3, 5]


def test_parse_page_selection_rejects_descending_range() -> None:
    with pytest.raises(PdfAnalyzeError, match="start page"):
        parse_page_selection("4-2", page_count=5)


def test_parse_page_selection_rejects_out_of_range_page() -> None:
    with pytest.raises(PdfAnalyzeError, match="exceed PDF page count"):
        parse_page_selection("1,6", page_count=5)


def test_pdf_analyze_tool_handler_missing_prompt(tmp_path: Path) -> None:
    tool = _build_pdf_analyze_tool(tmp_path)
    result = run(tool.handler(_invocation({"path": "report.pdf"})))

    assert result["resultType"] == "failure"
    assert "prompt" in result["textResultForLlm"].lower()


def test_pdf_analyze_tool_handler_success(tmp_path: Path) -> None:
    payload = _sample_payload(tmp_path / "report.pdf")

    with patch("copex.sdk_tools.prepare_pdf_analysis_payload", return_value=payload):
        tool = _build_pdf_analyze_tool(tmp_path)
        result = run(
            tool.handler(
                _invocation({"path": "report.pdf", "prompt": "Summarize the figures", "pages": "1-2"})
            )
        )

    assert result["resultType"] == "success"
    assert result["binaryResultsForLlm"] == payload.binary_results
    assert result["toolTelemetry"]["page_count"] == 2
    assert result["toolTelemetry"]["total_image_bytes"] == 12


def test_pdf_analyze_tool_handler_error(tmp_path: Path) -> None:
    with patch(
        "copex.sdk_tools.prepare_pdf_analysis_payload",
        side_effect=PdfAnalyzeError("not a PDF"),
    ):
        tool = _build_pdf_analyze_tool(tmp_path)
        result = run(tool.handler(_invocation({"path": "report.txt", "prompt": "Analyze"})))

    assert result["resultType"] == "failure"
    assert "not a PDF" in result["error"]


def test_pdf_screenshot_tool_handler_success(tmp_path: Path) -> None:
    written = [tmp_path / "renders" / "report-page-001.png", tmp_path / "renders" / "report-page-003.png"]

    with patch("copex.sdk_tools.export_pdf_screenshots", return_value=written):
        tool = _build_pdf_screenshot_tool(tmp_path)
        result = run(
            tool.handler(
                _invocation(
                    {"path": "report.pdf", "pages": "1,3", "output_dir": "renders"}
                )
            )
        )

    assert result["resultType"] == "success"
    assert "report-page-001.png" in result["textResultForLlm"]
    assert result["toolTelemetry"]["page_count"] == 2


def test_pdf_screenshot_tool_handler_missing_output_dir(tmp_path: Path) -> None:
    tool = _build_pdf_screenshot_tool(tmp_path)
    result = run(tool.handler(_invocation({"path": "report.pdf", "pages": "1"})))

    assert result["resultType"] == "failure"
    assert "output_dir" in result["textResultForLlm"]


def test_register_pdf_tools_with_pymupdf() -> None:
    with patch("copex.sdk_tools.pdf_support_available", return_value=True):
        assert register_pdf_tools() is True
    assert "pdf_analyze" in list_domain_tools()
    assert "pdf_screenshot" in list_domain_tools()


def test_register_pdf_tools_without_pymupdf() -> None:
    with patch("copex.sdk_tools.pdf_support_available", return_value=False):
        assert register_pdf_tools() is False


def test_register_pdf_tools_builds_tools(tmp_path: Path) -> None:
    with patch("copex.sdk_tools.pdf_support_available", return_value=True):
        register_pdf_tools()
    tools = build_domain_tools(["pdf_analyze", "pdf_screenshot"], working_dir=tmp_path)

    assert len(tools) == 2
    assert tools[0].name == "pdf_analyze"
    assert tools[1].name == "pdf_screenshot"


def test_config_pdf_analyze_default_false() -> None:
    assert CopexConfig().pdf_analyze is False


def test_config_pdf_analyze_enabled() -> None:
    assert CopexConfig(pdf_analyze=True).pdf_analyze is True


def test_config_to_session_options_includes_pdf_tools_when_enabled() -> None:
    config = CopexConfig(pdf_analyze=True)
    with patch("copex.sdk_tools.pdf_support_available", return_value=True):
        opts = config.to_session_options()

    assert "tools" in opts
    tool_names = [tool.name for tool in opts["tools"]]
    assert "pdf_analyze" in tool_names
    assert "pdf_screenshot" in tool_names


def test_config_to_session_options_no_pdf_tools_when_disabled() -> None:
    opts = CopexConfig(pdf_analyze=False).to_session_options()
    assert "tools" not in opts


def test_config_to_session_options_no_pdf_tools_when_dependency_missing() -> None:
    config = CopexConfig(pdf_analyze=True)
    with patch("copex.sdk_tools.pdf_support_available", return_value=False):
        opts = config.to_session_options()
    assert "tools" not in opts


def test_chat_pdf_analyze_flag_sets_config(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def _fake_run_chat(config, prompt, *_args, **_kwargs):
        captured["pdf_analyze"] = config.pdf_analyze
        captured["prompt"] = prompt

    monkeypatch.setattr(cli, "_run_chat", _fake_run_chat)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["chat", "--model", "gpt-4.1", "--pdf-analyze", "Analyze the PDF"])

    assert result.exit_code == 0
    assert captured["pdf_analyze"] is True
    assert captured["prompt"] == "Analyze the PDF"


def test_main_prompt_pdf_analyze_flag_sets_config(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def _fake_run_chat(config, prompt, *_args, **_kwargs):
        captured["pdf_analyze"] = config.pdf_analyze
        captured["prompt"] = prompt

    monkeypatch.setattr(cli, "_run_chat", _fake_run_chat)

    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["--model", "gpt-4.1", "-p", "Review this report", "--no-squad", "--pdf-analyze"],
    )

    assert result.exit_code == 0
    assert captured["pdf_analyze"] is True
    assert captured["prompt"] == "Review this report"


def test_prepare_pdf_analysis_payload_renders_real_pdf(tmp_path: Path) -> None:
    pytest.importorskip("fitz")
    pdf_path = tmp_path / "report.pdf"
    _create_tiny_pdf(pdf_path)

    payload = prepare_pdf_analysis_payload(
        "report.pdf",
        prompt="Summarize the charts and figures",
        working_dir=tmp_path,
        pages="1-2",
    )

    assert payload.display_path == "report.pdf"
    assert payload.selected_pages == (1, 2)
    assert len(payload.binary_results) == 2
    assert payload.binary_results[0]["mimeType"] == "image/png"
    assert base64.b64decode(payload.binary_results[0]["data"]).startswith(b"\x89PNG\r\n\x1a\n")
    assert "Summarize the charts and figures" in payload.prompt_text


def test_export_pdf_screenshots_writes_png_files(tmp_path: Path) -> None:
    pytest.importorskip("fitz")
    pdf_path = tmp_path / "report.pdf"
    _create_tiny_pdf(pdf_path)

    written_paths = export_pdf_screenshots(
        "report.pdf",
        pages="1,2",
        output_dir="renders",
        working_dir=tmp_path,
    )

    assert [path.name for path in written_paths] == ["report-page-001.png", "report-page-002.png"]
    assert all(path.exists() for path in written_paths)
    assert written_paths[0].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_prepare_pdf_analysis_payload_rejects_non_pdf(tmp_path: Path) -> None:
    pytest.importorskip("fitz")
    (tmp_path / "report.txt").write_text("not a pdf", encoding="utf-8")

    with pytest.raises(PdfAnalyzeError, match="(not a PDF|Failed to open PDF)"):
        prepare_pdf_analysis_payload(
            "report.txt",
            prompt="Analyze this file",
            working_dir=tmp_path,
        )


def test_export_pdf_screenshots_rejects_output_dir_outside_working_dir(tmp_path: Path) -> None:
    pytest.importorskip("fitz")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _create_tiny_pdf(workspace / "report.pdf")

    with pytest.raises(PdfAnalyzeError, match="Path traversal"):
        export_pdf_screenshots(
            "report.pdf",
            pages="1",
            output_dir="../outside",
            working_dir=workspace,
        )
