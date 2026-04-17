# 多文件类型支持设计方案

## 背景

当前 KB-AI 仅支持 `.md` 文件的知识库索引与检索。为了提升知识库覆盖率和实用性，需要扩展支持 **PDF、Word（.doc/.docx）、Excel（.xls/.xlsx）和图片（.png/.jpg 等）**，并确保现有混合检索（向量 + 关键词 + Rerank）的**召回率不下降**。

## 目标

1. 支持 PDF、Word、Excel、图片四类文件的解析与索引。
2. 所有新增文件类型的 chunk 与现有 `.md` chunk 共用同一套检索 pipeline（Milvus + PostgreSQL + RRF Fusion）。
3. 通过结构化解析、双文本合并、行级分块+摘要等策略，保证**召回率不降低**。
4. 第一阶段采用**文件级全量重索引**（快速上线），第二阶段再优化为增量更新。

## 非目标

- 第一阶段不实现增量更新（大文件改动时全量重索引）。
- 不直接支持视频、音频等其他富媒体文件。
- 不替换现有的 `.md` 处理逻辑。

---

## 架构设计

在现有的 `ingestion_service.py` 和检索 pipeline 之间新增**文档解析层（Document Parser Layer）**，职责是将磁盘上任意支持的文件类型统一解析为带元数据的 `Document` 列表，再走现有的分块 → 嵌入 → 索引流程。

```
文件路径
    ↓
FileTypeDetector
    ↓
PDFParser / WordParser / ExcelParser / ImageParser
    ↓
List[Document]（含统一 metadata）
    ↓
现有的 _hash_text → chunk → embed → Milvus / PostgreSQL
    ↓
现有的 混合检索 + Rerank → LLM 回答
```

### 核心模块

| 模块 | 职责 | 对应文件 |
|------|------|----------|
| `FileTypeDetector` | 根据扩展名判断文件类型 | `app/services/parsers/file_detector.py` |
| `PDFParser` | 结构化提取 PDF（标题、段落、表格、图片） | `app/services/parsers/pdf_parser.py` |
| `WordParser` | 结构化提取 Word（章节、段落、表格、图片） | `app/services/parsers/word_parser.py` |
| `ExcelParser` | 按 Sheet/行分块 + Sheet 摘要 | `app/services/parsers/excel_parser.py` |
| `ImageParser` | OCR 原文 + 多模态描述合并 | `app/services/parsers/image_parser.py` |
| `ParserPipeline` | 调度不同 Parser，统一输出 `List[Document]` | `app/services/parsers/__init__.py` |

所有 Parser 的输出必须包含统一的 `metadata`：

- `source`: 原始文件绝对路径
- `doc_type`: `pdf` / `word` / `excel` / `image`
- `page` / `sheet` / `row`: 位置信息（按类型填入）
- `element_type`: `text` / `table` / `image` / `summary`

新增 chunk 与现有 `.md` chunk 共用同一 Milvus collection 和 PostgreSQL `ChunkMetadata` 表，**检索端完全无感知**。

---

## 数据流设计

### 1. 通用入口

`ParserPipeline.parse_file(path: Path) -> List[Document]` 是 ingestion_service 唯一调用点。

### 2. PDF / Word 数据流

```
PDF/Word → Unstructured（或 Azure DI）解析
    ↓
提取 Element 列表：[Header, Paragraph, Table, ImageRef]
    ↓
相邻文本段按语义合并（避免单句一个 chunk）
    ↓
表格 → Markdown 格式文本（保留表头行）
图片 → 提取为临时文件 → 走 ImageParser → OCR+描述文本替换原 ImageRef 位置
    ↓
最终生成连续的 Document 列表，再走 RecursiveCharacterTextSplitter
```

### 3. Excel 数据流

```
Excel → pandas 读取每个 Sheet
    ↓
对每个 Sheet：
  ├─ 行级 chunk：
  │    page_content = "Sheet: {sheet_name}\n表头: {headers}\n行数据: {row_json}"
  │    metadata = {doc_type: "excel", sheet: "sheet_name", row_index: N}
  │
  └─ Sheet 摘要 chunk：
       整 Sheet 内容 → 调用 LLM 生成业务摘要
       metadata = {doc_type: "excel", sheet: "sheet_name", element_type: "summary"}
```

- 行级 chunk 保证精确查找（如"销售额是多少"）。
- Sheet 摘要 chunk 保证语义召回（如"各季度销售趋势"）。

### 4. 图片数据流

图片采用 **OCR + 多模态描述合并** 方案，兼顾字面匹配和语义匹配：

```
图片 → 并行调用：
  ├─ OCR API（如阿里灵骏文档智能） → text_ocr
  └─ 多模态 API（如 GPT-4o / Qwen-VL-Max） → text_caption
    ↓
合并为单个 Document：
  page_content = "[图片描述]\n{text_caption}\n[图片文字]\n{text_ocr}"
  metadata = {doc_type: "image", source: "原文件路径", element_type: "image"}
```

两段文本合并为**一个 chunk**，向量检索和关键词检索都能同时命中。

---

## 技术选型

| 文件类型 | 解析方案 | 依赖/服务 |
|----------|----------|-----------|
| PDF | `unstructured[pdf]`（首选）或 `Azure Document Intelligence`（备选） | `unstructured`, `pdf2image`, `pikepdf` |
| Word | `unstructured[docx]` | `unstructured`, `python-docx` |
| Excel | `pandas` + `openpyxl` | `pandas`, `openpyxl` |
| 图片 OCR | 阿里云/百度智能云文档智能 OCR API | `requests` |
| 图片描述 | GPT-4o Vision / 阿里云 Qwen-VL-Max API | `openai` SDK 或 `requests` |

**选型理由**：
- `unstructured` 对 PDF/Word 的标题层级、表格结构提取能力最强，召回率损失最小。
- `pandas` 处理 Excel 行级分块最轻量、最可控。
- 云端 OCR 和多模态 API 效果优于开源本地模型，无需额外 GPU 运维。

---

## 错误处理策略

| 场景 | 处理策略 |
|------|----------|
| 文件损坏 / 加密 PDF | Parser 抛出 `ParseError`，记录 error log，跳过该文件，不阻断批次内其他文件索引 |
| OCR / 多模态 API 超时 | 重试 3 次 + 指数退避（2s / 4s / 8s）。最终失败时降级为 `page_content="[图片解析失败] {filename}"`，保证文件其他部分正常索引 |
| Excel 超大文件（>10MB 或 >10万行） | 行级 chunk 改为采样前 5000 行 + 全量 Sheet 摘要，metadata 标记 `truncated: true` |
| Word 复杂格式（嵌套表格、文本框） | 跳过不可解析区域，记录 warning，继续处理可解析部分 |
| 图片内嵌在 PDF/Word 中 | 图片解析失败时保留占位符 `![图片: {filename}]`，避免上下文断裂 |

所有错误统一走现有日志系统，失败文件的路径和原因写入日志（未来可扩展写入 `ingestion_errors` 表）。

---

## 配置项

在 `app/core/config.py` 中新增以下配置（均有合理默认值）：

```python
# 图片解析
ENABLE_IMAGE_OCR: bool = True
ENABLE_IMAGE_CAPTION: bool = True
OCR_API_KEY: str = ""
OCR_API_ENDPOINT: str = ""
VISION_API_KEY: str = ""
VISION_MODEL: str = "gpt-4o"

# Excel 限制
EXCEL_MAX_ROWS_PER_SHEET: int = 5000
EXCEL_ENABLE_SHEET_SUMMARY: bool = True

# Parser 降级
PARSER_FALLBACK_TO_PLAINTEXT: bool = True  # 任何解析失败时是否尝试纯文本读取
```

---

## 测试策略

| 测试项 | 测试方式 |
|--------|----------|
| 单元测试：各 Parser | 每个 Parser 独立的 pytest 模块，使用真实样例文件验证输出结构、metadata 完整性 |
| 集成测试：端到端索引 | 上传混合类型文件 → 调用 ingest API → 验证 Milvus 和 PG 中 chunk 数量、metadata、hash 一致 |
| 召回率回归测试 | 对同一批文档（`.md` 导出版 vs 原生 PDF/Word 版）对比同一问题的 top-k 召回结果，确保召回率不下降 |
| Excel 行级检索测试 | 构造带表头的测试 Excel，针对特定行数据提问，验证目标行 chunk 出现在 top-5 |
| 图片合并文本测试 | mock OCR 和 caption API，验证最终 `page_content` 格式正确，关键词检索和向量检索均可命中 |

测试优先使用真实文件（API 调用部分可用 `vcr.py` 录制回放），符合项目规范。

---

## 实施阶段

### 第一阶段（MVP）

1. 搭建 `app/services/parsers/` 模块骨架和 `FileTypeDetector`。
2. 实现 `ExcelParser`（依赖最少，最容易验证）。
3. 实现 `ImageParser`（OCR + Caption 合并）。
4. 实现 `PDFParser` 和 `WordParser`（基于 `unstructured`）。
5. 改造 `ingestion_service.py`，用 `ParserPipeline` 替换现有的纯 `.md` 扫描逻辑。
6. 编写单元测试和集成测试。
7. 召回率回归验证。

### 第二阶段（优化）

1. 实现增量更新：文件级 hash 比对 → 仅对变更页/Sheet 重索引。
2. 支持更多图片格式（如 `.tiff`, `.webp`）。
3. 引入本地模型兜底选项（应对 API 不可用场景）。

---

## 风险与应对

| 风险 | 应对 |
|------|------|
| `unstructured` 安装依赖重 | 提供 `requirements-parsers.txt`，Docker 镜像预装系统依赖 |
| 图片 API 费用不可控 | 增加配置开关 + 单文件图片数量上限 |
| Excel 行级 chunk 爆炸 | 设置 `EXCEL_MAX_ROWS_PER_SHEET` 限制 |
| 召回率下降 | 通过结构保留、双文本合并、Sheet 摘要三重策略对冲，并用回归测试验证 |
