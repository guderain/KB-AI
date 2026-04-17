from pathlib import Path

import pandas as pd
import pytest

from app.core.config import get_settings
from app.services.parsers.excel_parser import ExcelParser


def test_excel_parser_outputs_row_chunks_and_summary():
    parser = ExcelParser()
    docs = parser.parse(Path("tests/fixtures/sample.xlsx"))

    row_docs = [d for d in docs if d.metadata.get("element_type") == "row"]
    summary_docs = [d for d in docs if d.metadata.get("element_type") == "summary"]

    assert len(row_docs) == 3
    assert len(summary_docs) == 1

    first = row_docs[0]
    assert "A产品" in first.page_content
    assert first.metadata["doc_type"] == "excel"
    assert first.metadata["sheet"] == "销售数据"
    assert first.metadata["row_index"] == 0

    summary = summary_docs[0]
    assert "销售数据" in summary.page_content
    assert "3 行数据" in summary.page_content


def test_excel_parser_truncates_rows(tmp_path, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "EXCEL_MAX_ROWS_PER_SHEET", 2)

    file_path = tmp_path / "large.xlsx"
    df = pd.DataFrame({
        "col": [f"val{i}" for i in range(5)]
    })
    df.to_excel(file_path, sheet_name="Sheet1", index=False)

    parser = ExcelParser()
    docs = parser.parse(file_path)

    row_docs = [d for d in docs if d.metadata.get("element_type") == "row"]
    summary_docs = [d for d in docs if d.metadata.get("element_type") == "summary"]

    assert len(row_docs) == 2
    assert len(summary_docs) == 1
    assert all(d.metadata["truncated"] is True for d in row_docs)
    assert summary_docs[0].metadata["truncated"] is True


def test_excel_parser_no_summary_when_disabled(tmp_path, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "EXCEL_ENABLE_SHEET_SUMMARY", False)

    file_path = tmp_path / "nosummary.xlsx"
    df = pd.DataFrame({"col": ["a", "b"]})
    df.to_excel(file_path, sheet_name="Sheet1", index=False)

    parser = ExcelParser()
    docs = parser.parse(file_path)

    row_docs = [d for d in docs if d.metadata.get("element_type") == "row"]
    summary_docs = [d for d in docs if d.metadata.get("element_type") == "summary"]

    assert len(row_docs) == 2
    assert len(summary_docs) == 0
