# 多文件类型支持实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 KB-AI RAG 系统中添加 PDF、Word、Excel、图片四种文件类型的解析与索引支持，确保召回率不下降。

**Architecture:** 新增 `app/services/parsers/` 文档解析层，统一将各类型文件转换为 `List[Document]`，再走现有的 chunk → embed → Milvus/PostgreSQL 流程。检索端完全无感知。

**Tech Stack:** FastAPI + LangChain + `unstructured` + `pandas/openpyxl` + 云端 OCR/Vision API + Milvus + PostgreSQL

---

## 文件结构总览

### 新增文件

| 文件 | 职责 |
|------|------|
| `app/services/parsers/__init__.py` | `ParserPipeline`：调度入口，统一输出 `List[Document]` |
| `app/services/parsers/file_detector.py` | `FileTypeDetector`：根据扩展名映射到对应 parser |
| `app/services/parsers/pdf_parser.py` | `PDFParser`：基于 `unstructured` 提取结构化文本、表格、内嵌图片 |
| `app/services/parsers/word_parser.py` | `WordParser`：基于 `unstructured` 提取章节、段落、表格、内嵌图片 |
| `app/services/parsers/excel_parser.py` | `ExcelParser`：pandas 读取，输出行级 chunk + Sheet 摘要 chunk |
| `app/services/parsers/image_parser.py` | `ImageParser`：OCR API + Vision API，合并输出单 chunk |
| `tests/services/parsers/test_file_detector.py` | FileTypeDetector 单元测试 |
| `tests/services/parsers/test_excel_parser.py` | ExcelParser 单元测试 |
| `tests/services/parsers/test_image_parser.py` | ImageParser 单元测试（mock API） |
| `tests/services/parsers/test_pdf_parser.py` | PDFParser 单元测试 |
| `tests/services/parsers/test_word_parser.py` | WordParser 单元测试 |
| `tests/fixtures/sample.xlsx` | Excel 测试样例 |
| `tests/fixtures/sample.docx` | Word 测试样例 |
| `tests/fixtures/sample.pdf` | PDF 测试样例 |

### 修改文件

| 文件 | 修改点 |
|------|--------|
| `app/core/config.py` | 新增图片 API、Excel 限制、Parser 降级等配置项 |
| `app/services/ingestion_service.py` | `_scan_md_files` → 通用扫描；`_load_file_doc` → `ParserPipeline.parse_file` |
| `requirements.txt` | 新增 `unstructured`, `pdf2image`, `pikepdf`, `python-docx`, `pandas`, `openpyxl`, `Pillow` |

---

## Task 1: 配置扩展与 FileTypeDetector

**Files:**
- Create: `app/services/parsers/__init__.py`
- Create: `app/services/parsers/file_detector.py`
- Modify: `app/core/config.py`
- Test: `tests/services/parsers/test_file_detector.py`

- [ ] **Step 1: 在 `app/core/config.py` 中新增 Parser 相关配置**

在 `Settings` 类中 `AUTO_INGEST_ON_STARTUP` 之后添加：

```python
    # Parser 配置
    ENABLE_IMAGE_OCR: bool = True
    ENABLE_IMAGE_CAPTION: bool = True
    OCR_API_KEY: str = ''
    OCR_API_ENDPOINT: str = ''
    VISION_API_KEY: str = ''
    VISION_API_ENDPOINT: str = ''
    VISION_MODEL: str = 'gpt-4o'

    EXCEL_MAX_ROWS_PER_SHEET: int = 5000
    EXCEL_ENABLE_SHEET_SUMMARY: bool = True

    PARSER_FALLBACK_TO_PLAINTEXT: bool = True
```

- [ ] **Step 2: 创建 `app/services/parsers/file_detector.py`**

```python
from enum import Enum, auto
from pathlib import Path


class FileType(Enum):
    MARKDOWN = auto()
    PDF = auto()
    WORD = auto()
    EXCEL = auto()
    IMAGE = auto()
    UNKNOWN = auto()


class FileTypeDetector:
    _ext_map: dict[str, FileType] = {
        '.md': FileType.MARKDOWN,
        '.pdf': FileType.PDF,
        '.doc': FileType.WORD,
        '.docx': FileType.WORD,
        '.xls': FileType.EXCEL,
        '.xlsx': FileType.EXCEL,
        '.png': FileType.IMAGE,
        '.jpg': FileType.IMAGE,
        '.jpeg': FileType.IMAGE,
        '.bmp': FileType.IMAGE,
        '.gif': FileType.IMAGE,
        '.webp': FileType.IMAGE,
    }

    @classmethod
    def detect(cls, path: Path) -> FileType:
        ext = path.suffix.lower()
        return cls._ext_map.get(ext, FileType.UNKNOWN)
```

- [ ] **Step 3: 编写测试 `tests/services/parsers/test_file_detector.py`**

```python
from pathlib import Path

from app.services.parsers.file_detector import FileType, FileTypeDetector


def test_detect_markdown():
    assert FileTypeDetector.detect(Path('doc.md')) == FileType.MARKDOWN


def test_detect_pdf():
    assert FileTypeDetector.detect(Path('doc.pdf')) == FileType.PDF


def test_detect_word_docx():
    assert FileTypeDetector.detect(Path('doc.docx')) == FileType.WORD


def test_detect_excel_xlsx():
    assert FileTypeDetector.detect(Path('data.xlsx')) == FileType.EXCEL


def test_detect_image_png():
    assert FileTypeDetector.detect(Path('img.png')) == FileType.IMAGE


def test_detect_unknown():
    assert FileTypeDetector.detect(Path('archive.zip')) == FileType.UNKNOWN


def test_detect_case_insensitive():
    assert FileTypeDetector.detect(Path('doc.PDF')) == FileType.PDF
    assert FileTypeDetector.detect(Path('img.JPG')) == FileType.IMAGE
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/services/parsers/test_file_detector.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add app/core/config.py app/services/parsers/file_detector.py tests/services/parsers/test_file_detector.py
git commit -m "feat: add parser configs and FileTypeDetector"
```

---

## Task 2: ExcelParser 实现与测试

**Files:**
- Create: `app/services/parsers/excel_parser.py`
- Create: `tests/fixtures/sample.xlsx`
- Create: `tests/services/parsers/test_excel_parser.py`
- Modify: `app/services/parsers/__init__.py` (注册)

- [ ] **Step 1: 创建 Excel 测试样例 `tests/fixtures/sample.xlsx`**

用 Python 脚本生成：

```python
import pandas as pd
from pathlib import Path

Path('tests/fixtures').mkdir(parents=True, exist_ok=True)
df = pd.DataFrame({
    '产品': ['A产品', 'B产品', 'C产品'],
    '销售额': [12000, 15000, 9000],
    '季度': ['Q1', 'Q1', 'Q1'],
})
with pd.ExcelWriter('tests/fixtures/sample.xlsx', engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='销售数据', index=False)
```

Run: `python -c "import pandas as pd; import os; os.makedirs('tests/fixtures', exist_ok=True); df=pd.DataFrame({'产品':['A产品','B产品','C产品'],'销售额':[12000,15000,9000],'季度':['Q1','Q1','Q1']}); df.to_excel('tests/fixtures/sample.xlsx', sheet_name='销售数据', index=False)"`

- [ ] **Step 2: 创建 `app/services/parsers/excel_parser.py`**

```python
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
        try:
            xls = pd.ExcelFile(path, engine='openpyxl')
        except Exception as exc:
            logger.warning('Failed to open Excel %s: %s', path, exc)
            return docs

        for sheet_name in xls.sheet_names:
            try:
                df = pd.read_excel(xls, sheet_name=sheet_name)
            except Exception as exc:
                logger.warning('Failed to read sheet %s in %s: %s', sheet_name, path, exc)
                continue

            df = df.where(pd.notnull(df), None)
            headers = list(df.columns)
            rows = df.to_dict(orient='records')

            max_rows = settings.EXCEL_MAX_ROWS_PER_SHEET
            truncated = False
            if len(rows) > max_rows:
                rows = rows[:max_rows]
                truncated = True

            header_str = ', '.join(str(h) for h in headers)
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
                            'source': str(path),
                            'doc_type': 'excel',
                            'sheet': sheet_name,
                            'row_index': idx,
                            'element_type': 'row',
                            'truncated': truncated,
                        },
                    )
                )

            if settings.EXCEL_ENABLE_SHEET_SUMMARY:
                summary_content = (
                    f"Sheet: {sheet_name}\n"
                    f"表头: {header_str}\n"
                    f"该 Sheet 共有 {len(rows)} 行数据（原始 {len(df)} 行）。"
                )
                docs.append(
                    Document(
                        page_content=summary_content,
                        metadata={
                            'source': str(path),
                            'doc_type': 'excel',
                            'sheet': sheet_name,
                            'element_type': 'summary',
                            'truncated': truncated,
                        },
                    )
                )

        return docs
```

- [ ] **Step 3: 编写测试 `tests/services/parsers/test_excel_parser.py`**

```python
from pathlib import Path

from app.services.parsers.excel_parser import ExcelParser


def test_excel_parser_outputs_row_chunks_and_summary():
    parser = ExcelParser()
    docs = parser.parse(Path('tests/fixtures/sample.xlsx'))

    row_docs = [d for d in docs if d.metadata.get('element_type') == 'row']
    summary_docs = [d for d in docs if d.metadata.get('element_type') == 'summary']

    assert len(row_docs) == 3
    assert len(summary_docs) == 1

    first = row_docs[0]
    assert 'A产品' in first.page_content
    assert first.metadata['doc_type'] == 'excel'
    assert first.metadata['sheet'] == '销售数据'
    assert first.metadata['row_index'] == 0

    summary = summary_docs[0]
    assert '销售数据' in summary.page_content
    assert '3 行数据' in summary.page_content
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/services/parsers/test_excel_parser.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add app/services/parsers/excel_parser.py tests/fixtures/sample.xlsx tests/services/parsers/test_excel_parser.py
git commit -m "feat: add ExcelParser with row-level chunks and sheet summaries"
```

---

## Task 3: ImageParser 实现与测试

**Files:**
- Create: `app/services/parsers/image_parser.py`
- Create: `tests/fixtures/sample.png`
- Create: `tests/services/parsers/test_image_parser.py`

- [ ] **Step 1: 创建 Image 测试样例 `tests/fixtures/sample.png`**

```python
from PIL import Image
import os

os.makedirs('tests/fixtures', exist_ok=True)
img = Image.new('RGB', (100, 30), color='white')
img.save('tests/fixtures/sample.png')
```

Run: `python -c "from PIL import Image; import os; os.makedirs('tests/fixtures', exist_ok=True); img=Image.new('RGB', (100,30), color='white'); img.save('tests/fixtures/sample.png')"`

- [ ] **Step 2: 创建 `app/services/parsers/image_parser.py`**

```python
import base64
import logging
from pathlib import Path

from langchain_core.documents import Document

from app.core.config import get_settings
from app.services.dependencies import get_async_http_client

logger = logging.getLogger(__name__)


class ImageParser:
    def parse(self, path: Path) -> list[Document]:
        settings = get_settings()
        if not settings.ENABLE_IMAGE_OCR and not settings.ENABLE_IMAGE_CAPTION:
            return []

        mime = self._guess_mime(path)
        try:
            b64 = self._encode_image(path)
        except Exception as exc:
            logger.warning('Failed to read image %s: %s', path, exc)
            return self._fallback_doc(path)

        text_ocr = ''
        text_caption = ''

        if settings.ENABLE_IMAGE_OCR and settings.OCR_API_ENDPOINT:
            try:
                text_ocr = self._call_ocr(b64, mime)
            except Exception as exc:
                logger.warning('OCR failed for %s: %s', path, exc)

        if settings.ENABLE_IMAGE_CAPTION and settings.VISION_API_ENDPOINT:
            try:
                text_caption = self._call_caption(b64, mime)
            except Exception as exc:
                logger.warning('Vision caption failed for %s: %s', path, exc)

        if not text_ocr and not text_caption:
            return self._fallback_doc(path)

        page_content = ''
        if text_caption:
            page_content += f'[图片描述]\n{text_caption}\n'
        if text_ocr:
            page_content += f'[图片文字]\n{text_ocr}'

        return [
            Document(
                page_content=page_content.strip(),
                metadata={
                    'source': str(path),
                    'doc_type': 'image',
                    'element_type': 'image',
                },
            )
        ]

    def _guess_mime(self, path: Path) -> str:
        suffix = path.suffix.lower()
        mapping = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.bmp': 'image/bmp',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
        }
        return mapping.get(suffix, 'image/jpeg')

    def _encode_image(self, path: Path) -> str:
        return base64.b64encode(path.read_bytes()).decode('utf-8')

    def _fallback_doc(self, path: Path) -> list[Document]:
        return [
            Document(
                page_content=f'[图片解析失败] {path.name}',
                metadata={
                    'source': str(path),
                    'doc_type': 'image',
                    'element_type': 'image',
                },
            )
        ]

    def _call_ocr(self, b64: str, mime: str) -> str:
        """Placeholder: 实际 OCR API 调用在 Task 3.5 中替换为真实实现。"""
        return ''

    def _call_caption(self, b64: str, mime: str) -> str:
        """Placeholder: 实际 Vision API 调用在 Task 3.5 中替换为真实实现。"""
        return ''
```

- [ ] **Step 3: 编写测试 `tests/services/parsers/test_image_parser.py`**

```python
from pathlib import Path

from app.services.parsers.image_parser import ImageParser


def test_image_parser_fallback_when_no_apis():
    parser = ImageParser()
    docs = parser.parse(Path('tests/fixtures/sample.png'))
    assert len(docs) == 1
    assert '[图片解析失败]' in docs[0].page_content
    assert docs[0].metadata['doc_type'] == 'image'


def test_image_parser_merge_ocr_and_caption(monkeypatch):
    parser = ImageParser()

    def mock_ocr(self, b64, mime):
        return 'OCR text here'

    def mock_caption(self, b64, mime):
        return 'Caption text here'

    monkeypatch.setattr(ImageParser, '_call_ocr', mock_ocr)
    monkeypatch.setattr(ImageParser, '_call_caption', mock_caption)

    docs = parser.parse(Path('tests/fixtures/sample.png'))
    assert len(docs) == 1
    content = docs[0].page_content
    assert '[图片描述]' in content
    assert 'Caption text here' in content
    assert '[图片文字]' in content
    assert 'OCR text here' in content
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/services/parsers/test_image_parser.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add app/services/parsers/image_parser.py tests/fixtures/sample.png tests/services/parsers/test_image_parser.py
git commit -m "feat: add ImageParser with OCR + caption merge and fallback"
```

---

## Task 4: PDFParser 与 WordParser 实现与测试

**Files:**
- Create: `app/services/parsers/pdf_parser.py`
- Create: `app/services/parsers/word_parser.py`
- Create: `tests/fixtures/sample.pdf`
- Create: `tests/fixtures/sample.docx`
- Create: `tests/services/parsers/test_pdf_parser.py`
- Create: `tests/services/parsers/test_word_parser.py`

- [ ] **Step 1: 安装 unstructured 依赖**

Run: `pip install unstructured pdf2image pikepdf python-docx`

然后更新 `requirements.txt`，在末尾添加：

```text
unstructured==0.16.0
pdf2image==1.17.0
pikepdf==9.0.0
python-docx==1.1.2
pandas==2.2.3
openpyxl==3.1.5
Pillow==11.0.0
```

- [ ] **Step 2: 创建测试样例 `tests/fixtures/sample.docx` 和 `tests/fixtures/sample.pdf`**

`sample.docx`:
```python
from docx import Document as DocxDocument
import os

os.makedirs('tests/fixtures', exist_ok=True)
doc = DocxDocument()
doc.add_heading('第一章 简介', level=1)
doc.add_paragraph('这是简介段落。')
doc.add_heading('1.1 背景', level=2)
doc.add_paragraph('背景介绍内容。')
table = doc.add_table(rows=2, cols=2)
table.rows[0].cells[0].text = '名称'
table.rows[0].cells[1].text = '值'
table.rows[1].cells[0].text = 'A'
table.rows[1].cells[1].text = '100'
doc.save('tests/fixtures/sample.docx')
```

Run: `python -c "from docx import Document as DocxDocument; import os; os.makedirs('tests/fixtures', exist_ok=True); doc=DocxDocument(); doc.add_heading('第一章 简介', level=1); doc.add_paragraph('这是简介段落。'); doc.add_heading('1.1 背景', level=2); doc.add_paragraph('背景介绍内容。'); table=doc.add_table(rows=2, cols=2); table.rows[0].cells[0].text='名称'; table.rows[0].cells[1].text='值'; table.rows[1].cells[0].text='A'; table.rows[1].cells[1].text='100'; doc.save('tests/fixtures/sample.docx')"`

`sample.pdf` 通过把 docx 转成 pdf（需要 LibreOffice 或直接用 `fpdf` 生成简单 PDF）：
```python
from fpdf import FPDF
import os

os.makedirs('tests/fixtures', exist_ok=True)
pdf = FPDF()
pdf.add_page()
pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
pdf.set_font('DejaVu', size=12)
pdf.cell(200, 10, txt='第一章 简介', ln=True)
pdf.cell(200, 10, txt='这是简介段落。', ln=True)
pdf.output('tests/fixtures/sample.pdf')
```

更简单的方式：直接用一个已有的空白 PDF 或者纯英文 PDF。如果没有 `fpdf`，用 `reportlab`：

Run: `pip install reportlab`
Run: `python -c "from reportlab.pdfgen import canvas; import os; os.makedirs('tests/fixtures', exist_ok=True); c=canvas.Canvas('tests/fixtures/sample.pdf'); c.drawString(100,700,'Chapter 1 Introduction'); c.drawString(100,680,'This is an introduction paragraph.'); c.save()"`

- [ ] **Step 3: 创建 `app/services/parsers/pdf_parser.py`**

```python
import logging
from pathlib import Path

from langchain_core.documents import Document
from unstructured.partition.pdf import partition_pdf

logger = logging.getLogger(__name__)


class PDFParser:
    def parse(self, path: Path) -> list[Document]:
        docs: list[Document] = []
        try:
            elements = partition_pdf(
                filename=str(path),
                strategy='hi_res',
                infer_table_structure=True,
                extract_images_in_pdf=False,
            )
        except Exception as exc:
            logger.warning('Failed to parse PDF %s: %s', path, exc)
            return docs

        for idx, element in enumerate(elements):
            text = str(element)
            if not text.strip():
                continue
            metadata = {
                'source': str(path),
                'doc_type': 'pdf',
                'page': getattr(element, 'metadata', {}).get('page_number', 1),
                'element_type': getattr(element, 'category', 'text'),
            }
            docs.append(Document(page_content=text, metadata=metadata))

        return docs
```

- [ ] **Step 4: 创建 `app/services/parsers/word_parser.py`**

```python
import logging
from pathlib import Path

from langchain_core.documents import Document
from unstructured.partition.docx import partition_docx

logger = logging.getLogger(__name__)


class WordParser:
    def parse(self, path: Path) -> list[Document]:
        docs: list[Document] = []
        try:
            elements = partition_docx(filename=str(path))
        except Exception as exc:
            logger.warning('Failed to parse Word %s: %s', path, exc)
            return docs

        for element in elements:
            text = str(element)
            if not text.strip():
                continue
            metadata = {
                'source': str(path),
                'doc_type': 'word',
                'element_type': getattr(element, 'category', 'text'),
            }
            docs.append(Document(page_content=text, metadata=metadata))

        return docs
```

- [ ] **Step 5: 编写测试 `tests/services/parsers/test_pdf_parser.py`**

```python
from pathlib import Path

from app.services.parsers.pdf_parser import PDFParser


def test_pdf_parser_extracts_text():
    parser = PDFParser()
    docs = parser.parse(Path('tests/fixtures/sample.pdf'))
    assert len(docs) >= 1
    combined = ' '.join(d.page_content for d in docs)
    assert 'Chapter 1' in combined or '第一章' in combined
    assert all(d.metadata['doc_type'] == 'pdf' for d in docs)
```

- [ ] **Step 6: 编写测试 `tests/services/parsers/test_word_parser.py`**

```python
from pathlib import Path

from app.services.parsers.word_parser import WordParser


def test_word_parser_extracts_text_and_table():
    parser = WordParser()
    docs = parser.parse(Path('tests/fixtures/sample.docx'))
    combined = ' '.join(d.page_content for d in docs)
    assert '第一章 简介' in combined
    assert '背景介绍内容' in combined
    assert all(d.metadata['doc_type'] == 'word' for d in docs)
```

- [ ] **Step 7: 运行测试验证通过**

Run: `pytest tests/services/parsers/test_pdf_parser.py tests/services/parsers/test_word_parser.py -v`
Expected: 2 passed

- [ ] **Step 8: Commit**

```bash
git add app/services/parsers/pdf_parser.py app/services/parsers/word_parser.py tests/fixtures/sample.pdf tests/fixtures/sample.docx tests/services/parsers/test_pdf_parser.py tests/services/parsers/test_word_parser.py requirements.txt
git commit -m "feat: add PDFParser and WordParser based on unstructured"
```

---

## Task 5: ParserPipeline 与 Ingestion Service 改造

**Files:**
- Modify: `app/services/parsers/__init__.py`
- Modify: `app/services/ingestion_service.py`
- Test: `tests/services/parsers/test_pipeline.py`

- [ ] **Step 1: 实现 `ParserPipeline`（`app/services/parsers/__init__.py`）**

```python
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from app.services.parsers.excel_parser import ExcelParser
from app.services.parsers.file_detector import FileType, FileTypeDetector
from app.services.parsers.image_parser import ImageParser
from app.services.parsers.pdf_parser import PDFParser
from app.services.parsers.word_parser import WordParser


class ParserPipeline:
    def __init__(self) -> None:
        self._detector = FileTypeDetector()
        self._parsers = {
            FileType.EXCEL: ExcelParser(),
            FileType.IMAGE: ImageParser(),
            FileType.PDF: PDFParser(),
            FileType.WORD: WordParser(),
        }

    def parse_file(self, path: Path) -> list[Document]:
        file_type = self._detector.detect(path)

        if file_type == FileType.MARKDOWN or file_type == FileType.UNKNOWN:
            return self._load_plaintext(path)

        parser = self._parsers.get(file_type)
        if parser is None:
            return self._load_plaintext(path)

        return parser.parse(path)

    def _load_plaintext(self, path: Path) -> list[Document]:
        loader = TextLoader(str(path), encoding='utf-8')
        docs = loader.load()
        if not docs:
            return [Document(page_content='', metadata={'source': str(path)})]
        return docs
```

- [ ] **Step 2: 改造 `app/services/ingestion_service.py`**

将 `_scan_md_files` 改名为 `_scan_files`，扩展支持的扩展名列表：

```python
def _scan_files(root: str) -> list[Path]:
    supported_exts = {'.md', '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'}
    results: list[Path] = []
    for p in Path(root).rglob('*'):
        if p.is_file() and p.suffix.lower() in supported_exts:
            results.append(p)
    return sorted(results)
```

将 `_load_file_doc` 替换为使用 `ParserPipeline`：

```python
from app.services.parsers import ParserPipeline

_pipeline = ParserPipeline()


def _load_file_docs(path: Path) -> list[Document]:
    return _pipeline.parse_file(path)
```

修改 `reindex()` 和 `incremental_reindex()` 中关于文件扫描和 hash 的逻辑：

在 `reindex()` 中：
```python
    files = _scan_files(settings.KNOWLEDGE_BASE_DIR)
    docs: list[Document] = []
    file_hash_map: dict[str, str] = {}

    for path in files:
        file_docs = _load_file_docs(path)
        docs.extend(file_docs)
        combined_text = ''.join(d.page_content for d in file_docs)
        file_hash_map[str(path)] = _hash_text(combined_text)
```

在 `incremental_reindex()` 中同样修改扫描逻辑：
```python
    files = _scan_files(settings.KNOWLEDGE_BASE_DIR)
    current_docs_map: dict[str, list[Document]] = {}
    current_hashes: dict[str, str] = {}

    for path in files:
        file_docs = _load_file_docs(path)
        source = str(path)
        current_docs_map[source] = file_docs
        combined_text = ''.join(d.page_content for d in file_docs)
        current_hashes[source] = _hash_text(combined_text)
```

以及后续 `changed_docs` 的生成：
```python
        changed_docs: list[Document] = []
        for source in changed_sources:
            changed_docs.extend(current_docs_map[source])
```

还有 `ensure_index_ready_on_startup()` 中的扫描：
```python
    files = _scan_files(settings.KNOWLEDGE_BASE_DIR)
```

- [ ] **Step 3: 编写集成测试 `tests/services/parsers/test_pipeline.py`**

```python
from pathlib import Path

from app.services.parsers import ParserPipeline


def test_pipeline_parses_excel():
    pipeline = ParserPipeline()
    docs = pipeline.parse_file(Path('tests/fixtures/sample.xlsx'))
    assert any(d.metadata['doc_type'] == 'excel' for d in docs)


def test_pipeline_parses_image():
    pipeline = ParserPipeline()
    docs = pipeline.parse_file(Path('tests/fixtures/sample.png'))
    assert len(docs) == 1
    assert docs[0].metadata['doc_type'] == 'image'


def test_pipeline_parses_pdf():
    pipeline = ParserPipeline()
    docs = pipeline.parse_file(Path('tests/fixtures/sample.pdf'))
    assert any(d.metadata['doc_type'] == 'pdf' for d in docs)


def test_pipeline_parses_word():
    pipeline = ParserPipeline()
    docs = pipeline.parse_file(Path('tests/fixtures/sample.docx'))
    assert any(d.metadata['doc_type'] == 'word' for d in docs)


def test_pipeline_fallback_for_plaintext():
    pipeline = ParserPipeline()
    docs = pipeline.parse_file(Path('tests/services/parsers/test_pipeline.py'))
    assert len(docs) >= 1
    assert docs[0].metadata.get('source', '').endswith('test_pipeline.py')
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/services/parsers/test_pipeline.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add app/services/parsers/__init__.py app/services/ingestion_service.py tests/services/parsers/test_pipeline.py
git commit -m "feat: wire ParserPipeline into ingestion_service for multi-filetype support"
```

---

## Task 6: 端到端集成与回归测试

**Files:**
- Create: `tests/services/test_ingestion_multi_filetype.py`
- Modify: 现有 `app/services/ingestion_service.py`（如有遗漏）

- [ ] **Step 1: 编写端到端测试 `tests/services/test_ingestion_multi_filetype.py`**

该测试验证 `reindex()` 能够正确索引混合类型文件。

```python
import os
import tempfile

from app.core.config import get_settings
from app.db.postgres import SessionLocal
from app.models.doc_index import DocIndex
from app.services.ingestion_service import reindex


def test_reindex_ingests_excel_and_image():
    settings = get_settings()
    original_dir = settings.KNOWLEDGE_BASE_DIR

    with tempfile.TemporaryDirectory() as tmpdir:
        # 复制测试 fixture 到临时目录
        import shutil
        for name in ['sample.xlsx', 'sample.png']:
            shutil.copy(f'tests/fixtures/{name}', os.path.join(tmpdir, name))

        # 临时修改配置（只影响当前线程内的 settings 是不可行的，因为 get_settings 有 lru_cache）
        # 更好的方式是直接调用底层函数或 fixture，但这里简化：通过环境变量或 monkeypatch
        # 实际测试中建议用 monkeypatch 修改 settings.KNOWLEDGE_BASE_DIR
        # 由于 lru_cache，这里用 monkeypatch 修改 settings 对象的属性
        settings.KNOWLEDGE_BASE_DIR = tmpdir

        try:
            files_count, chunks_count = reindex()
            assert files_count == 2
            assert chunks_count > 0

            db = SessionLocal()
            try:
                rows = db.query(DocIndex).all()
                sources = {r.source for r in rows}
                assert any('sample.xlsx' in s for s in sources)
                assert any('sample.png' in s for s in sources)
            finally:
                db.close()
        finally:
            settings.KNOWLEDGE_BASE_DIR = original_dir
```

- [ ] **Step 2: 运行测试（需要本地 Milvus 和 PostgreSQL 运行）**

如果本地 infra 未启动，先启动：
Run: `docker-compose up -d`

然后运行测试：
Run: `pytest tests/services/test_ingestion_multi_filetype.py -v`
Expected: 1 passed（假设数据库和 Milvus 可连接）

- [ ] **Step 3: Commit**

```bash
git add tests/services/test_ingestion_multi_filetype.py
git commit -m "test: add end-to-end ingestion test for multi-filetype support"
```

---

## Task 7: Image API 真实调用实现（可选，云端对接）

**Files:**
- Modify: `app/services/parsers/image_parser.py`

- [ ] **Step 1: 实现 `_call_ocr`（以阿里云通用文字识别为例）**

```python
    def _call_ocr(self, b64: str, mime: str) -> str:
        settings = get_settings()
        import httpx
        body = {
            'image': b64,
        }
        resp = httpx.post(
            settings.OCR_API_ENDPOINT,
            json=body,
            headers={'Authorization': f'Bearer {settings.OCR_API_KEY}'},
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        # 根据实际阿里云返回格式调整
        return data.get('content', '')
```

- [ ] **Step 2: 实现 `_call_caption`（以 OpenAI 兼容 Vision API 为例）**

```python
    def _call_caption(self, b64: str, mime: str) -> str:
        settings = get_settings()
        import httpx
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': '请用中文简要描述这张图片的内容和场景。'},
                    {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{b64}'}},
                ],
            }
        ]
        resp = httpx.post(
            f'{settings.VISION_API_ENDPOINT}/chat/completions',
            json={'model': settings.VISION_MODEL, 'messages': messages, 'max_tokens': 256},
            headers={'Authorization': f'Bearer {settings.VISION_API_KEY}'},
            timeout=45.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return data['choices'][0]['message']['content']
```

- [ ] **Step 3: Commit**

```bash
git add app/services/parsers/image_parser.py
git commit -m "feat: integrate real OCR and Vision APIs in ImageParser"
```

---

## 验收检查清单

- [ ] `FileTypeDetector` 能正确识别所有 4 类文件 + Markdown
- [ ] `ExcelParser` 输出行级 chunk 和 Sheet 摘要 chunk，metadata 完整
- [ ] `ImageParser` 在无 API 时降级，有 API 时正确合并 OCR + caption
- [ ] `PDFParser` / `WordParser` 基于 `unstructured` 提取文本和表格
- [ ] `ParserPipeline` 能统一调度所有文件类型
- [ ] `ingestion_service.py` 的扫描、hash、索引逻辑支持多文件类型
- [ ] 端到端测试通过（需要本地 Milvus + PostgreSQL）
- [ ] `requirements.txt` 包含所有新增依赖
- [ ] `.env` 中已补充图片 API 相关配置（文档说明）
