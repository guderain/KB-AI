import json
import logging
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class ExcelParser:
    def parse(self, path: Path) -> list[Document]:
        settings = get_settings()
        docs: list[Document] = []

        # Choose engine based on file extension: xlrd for legacy .xls, openpyxl otherwise
        engine = "xlrd" if path.suffix.lower() == ".xls" else "openpyxl"
        try:
            xls = pd.ExcelFile(path, engine=engine)
        except Exception as exc:
            logger.warning("Failed to open Excel %s: %s", path, exc)
            return docs

        max_rows = settings.EXCEL_MAX_ROWS_PER_SHEET

        for sheet_name in xls.sheet_names:
            try:
                # Determine original row count without loading all data into memory
                if engine == "openpyxl":
                    ws = xls.book[sheet_name]
                    original_row_count = max(0, ws.max_row - 1)  # minus header
                else:
                    sheet = xls.book.sheet_by_name(sheet_name)
                    original_row_count = max(0, sheet.nrows - 1)  # minus header

                truncated = original_row_count > max_rows
                df = pd.read_excel(xls, sheet_name=sheet_name, nrows=max_rows)
            except Exception as exc:
                logger.warning("Failed to read sheet %s in %s: %s", sheet_name, path, exc)
                continue

            df = df.where(pd.notnull(df), None)
            headers = list(df.columns)
            rows = df.to_dict(orient="records")

            header_str = ", ".join(str(h) for h in headers)
            for idx, row in enumerate(rows):
                row_json = json.dumps(row, ensure_ascii=False)
                page_content = (
                    f"Sheet: {sheet_name}\n"
                    f"表头: {header_str}\n"
                    f"行数据: {row_json}"
                )
                docs.append(
                    Document(
                        page_content=page_content,
                        metadata={
                            "source": str(path),
                            "doc_type": "excel",
                            "sheet": sheet_name,
                            "row_index": idx,
                            "element_type": "row",
                            "truncated": truncated,
                        },
                    )
                )

            if settings.EXCEL_ENABLE_SHEET_SUMMARY:
                summary_content = (
                    f"Sheet: {sheet_name}\n"
                    f"表头: {header_str}\n"
                    f"该 Sheet 共有 {len(rows)} 行数据（原始 {original_row_count} 行）。"
                )
                docs.append(
                    Document(
                        page_content=summary_content,
                        metadata={
                            "source": str(path),
                            "doc_type": "excel",
                            "sheet": sheet_name,
                            "element_type": "summary",
                            "truncated": truncated,
                        },
                    )
                )

        return docs
