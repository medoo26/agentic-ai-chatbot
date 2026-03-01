# document_processor.py
import os
import re
import inspect
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Any

from pypdf import PdfReader
from docx import Document as DocxDocument


class DocumentProcessor:
    def __init__(self, rag_system, llm_service=None):
        self.rag_system = rag_system
        self.llm_service = llm_service

        # OCR optional (default OFF)
        self.enable_ocr = (os.getenv("ENABLE_OCR", "0").strip() == "1")
        self.tesseract_cmd = (os.getenv("TESSERACT_CMD") or "").strip()

        # Folder for converted HTML (stored as .txt)
        processed_dir = (os.getenv("PROCESSED_HTML_DIR") or "processed_html").strip()
        self.processed_html_dir = Path(processed_dir)
        self.processed_html_dir.mkdir(parents=True, exist_ok=True)

        # HTML chunking settings (fallback only)
        self.chunk_size = int(os.getenv("HTML_CHUNK_SIZE", "1800"))
        self.chunk_overlap = int(os.getenv("HTML_CHUNK_OVERLAP", "350"))

        # LLM conversion protection
        self.html_llm_timeout = int(os.getenv("HTML_LLM_TIMEOUT", "60"))

        # ✅ حد نهائي كبير فقط
        self.html_hard_max_chars = int(os.getenv("HTML_HARD_MAX_CHARS", "250000"))

        # Segment large docs for LLM conversion
        self.html_segment_chars = int(os.getenv("HTML_SEGMENT_CHARS", "12000"))
        self.html_segment_overlap = int(os.getenv("HTML_SEGMENT_OVERLAP", "400"))

        # Merge tiny chunks so RAG doesn't retrieve headings only
        self.min_chunk_chars = int(os.getenv("HTML_MIN_CHUNK_CHARS", "250"))

    # =========================================================
    # Helpers: clean stored filename -> original filename
    # =========================================================
    def _clean_original_name(self, stored_basename: str) -> str:
        name = (stored_basename or "").strip()
        if not name:
            return ""

        m = re.match(
            r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}_(.+)$",
            name,
        )
        if m:
            return (m.group(1) or "").strip()

        parts = name.split("_", 1)
        if len(parts) == 2 and len(parts[0]) >= 20:
            return parts[1].strip()

        return name

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    # =========================================================
    # Save converted HTML (as .txt)
    # =========================================================
    def _save_converted_html_as_txt(self, original_filename: str, html: str) -> Tuple[str, str]:
        stem = Path(original_filename).stem
        out_name = f"{stem}__html.txt"
        out_path = self.processed_html_dir / out_name
        out_path.write_text(html or "", encoding="utf-8")
        return str(out_path), out_name

    # =========================================================
    # ✅ Split long raw text into segments for LLM conversion
    # =========================================================
    def _split_text_segments(self, text: str) -> List[str]:
        t = (text or "").strip()
        if not t:
            return []

        seg = max(1500, self.html_segment_chars)
        ov = max(0, min(self.html_segment_overlap, seg - 1))
        step = max(1, seg - ov)

        if len(t) <= seg:
            return [t]

        out: List[str] = []
        for i in range(0, len(t), step):
            part = t[i : i + seg].strip()
            if part:
                out.append(part)
            if i + seg >= len(t):
                break
        return out

    # =========================================================
    # ✅ Robust HTML body extraction (prevents multiple doctypes)
    # =========================================================
    def _strip_outer_html(self, html: str) -> str:
        h = (html or "").strip()
        if not h:
            return ""

        # remove doctype
        h = re.sub(r"(?is)<!doctype[^>]*>", "", h).strip()
        # remove <html ...> and </html>
        h = re.sub(r"(?is)<html\b[^>]*>", "", h).strip()
        h = re.sub(r"(?is)</html\s*>", "", h).strip()
        # remove <head ...>...</head>
        h = re.sub(r"(?is)<head\b[^>]*>.*?</head\s*>", "", h).strip()

        # if <body> exists -> take inside
        m = re.search(r"(?is)<body\b[^>]*>(.*?)</body\s*>", h)
        if m:
            return (m.group(1) or "").strip()

        # else remove body tags if present but broken
        h = re.sub(r"(?is)<body\b[^>]*>", "", h).strip()
        h = re.sub(r"(?is)</body\s*>", "", h).strip()
        return h.strip()

    def _wrap_html_document(self, body_inner: str, title: str) -> str:
        t = (title or "Document").strip() or "Document"
        b = (body_inner or "").strip()
        return f"""<!doctype html>
<html lang="ar">
<head>
  <meta charset="utf-8">
  <title>{t}</title>
</head>
<body>
{b}
</body>
</html>
""".strip()

    # =========================================================
    # ✅ HTML -> plain text (for embeddings)
    # =========================================================
    def _html_to_plain(self, html: str) -> str:
        x = (html or "").strip()
        if not x:
            return ""
        x = re.sub(r"(?is)<br\s*/?>", "\n", x)
        x = re.sub(r"(?is)</p\s*>", "\n\n", x)
        x = re.sub(r"(?is)</h[1-6]\s*>", "\n", x)
        x = re.sub(r"(?is)<[^>]+>", " ", x)
        x = re.sub(r"[ \t]+", " ", x)
        x = re.sub(r"\n{3,}", "\n\n", x)
        return x.strip()

    # =========================================================
    # ✅ Generic HTML split (works for ANY docs)
    # - If many headings exist, split by headings
    # - else fallback to Recursive splitter
    # =========================================================
    def _split_html(self, html_doc: str) -> List[str]:
        html_doc = (html_doc or "").strip()
        if not html_doc:
            return []

        body = self._strip_outer_html(html_doc)

        # split by headings if present
        heading_pat = re.compile(r"(?is)(<h[1-6]\b[^>]*>.*?</h[1-6]\s*>)")
        parts = heading_pat.split(body)

        chunks: List[str] = []

        # parts: [before, h, after, h, after ...]
        if len(parts) > 1:
            before = (parts[0] or "").strip()
            if before:
                chunks.append(before)

            i = 1
            while i < len(parts) - 1:
                h = (parts[i] or "").strip()
                b = (parts[i + 1] or "").strip()
                block = (h + "\n" + b).strip()
                if block:
                    chunks.append(block)
                i += 2

            # if splitting produced enough segments, keep it
            if len(chunks) >= 5:
                return self._merge_tiny_text_chunks(chunks)

        # fallback: langchain splitter
        chunks2: List[str] = []
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", "• ", "- ", "؛", "؟", "!", ".", "،", " ", ""],
            )
            chunks2 = [c.strip() for c in splitter.split_text(body) if (c or "").strip()]
        except Exception:
            step = max(1, self.chunk_size - self.chunk_overlap)
            for j in range(0, len(body), step):
                part = body[j : j + self.chunk_size].strip()
                if part:
                    chunks2.append(part)

        return self._merge_tiny_text_chunks(chunks2)

    def _merge_tiny_text_chunks(self, chunks: List[str]) -> List[str]:
        merged: List[str] = []
        buf = ""
        for c in chunks:
            c = (c or "").strip()
            if not c:
                continue
            if not buf:
                buf = c
                continue
            if len(buf) < self.min_chunk_chars:
                buf = (buf + "\n\n" + c).strip()
                continue
            merged.append(buf)
            buf = c
        if buf:
            merged.append(buf)
        return [m for m in merged if (m or "").strip()]

    # =========================================================
    # Delete old indexed chunks for the same doc_key (important!)
    # =========================================================
    def _delete_existing_index(self, doc_key: str) -> None:
        dk = (doc_key or "").strip()
        if not dk:
            return

        rag = self.rag_system
        if rag is None:
            return

        try:
            coll = getattr(rag, "collection", None)
            if coll is not None:
                coll.delete(where={"doc_key": dk})
                print(f"🧹 Deleted previous indexed chunks where doc_key={dk}")
                return
        except Exception as e:
            print("⚠️ Could not delete previous chunks by doc_key:", e)

        try:
            fn = getattr(rag, "delete_doc_key", None)
            if callable(fn):
                fn(dk)
                print(f"🧹 Deleted previous indexed chunks via rag_system.delete_doc_key({dk})")
        except Exception as e:
            print("⚠️ Fallback delete_doc_key failed:", e)

    # =========================================================
    # Main processor
    # =========================================================
    async def process_file(self, file_path: str) -> Dict[str, Any]:
        file_ext = os.path.splitext(file_path)[1].lower()

        try:
            content = ""
            tables_md: List[str] = []

            if file_ext in [".png", ".jpg", ".jpeg"]:
                content = self._process_image(file_path)
            elif file_ext == ".pdf":
                content = self._process_pdf(file_path)
            elif file_ext in [".docx", ".doc"]:
                content, tables_md = self._process_docx_with_tables(file_path)
            elif file_ext == ".txt":
                content = self._process_txt(file_path)
            else:
                raise ValueError(f"نوع الملف {file_ext} غير مدعوم")

            content = (content or "").strip()
            tables_md = [t.strip() for t in (tables_md or []) if (t or "").strip()]

            if not content and not tables_md:
                print("⚠️ الملف فارغ بعد المعالجة")
                return {"html": "", "converted_txt_path": "", "converted_name": ""}

            stored_filename = os.path.basename(file_path)
            original_name = self._clean_original_name(stored_filename)

            doc_key = (original_name or stored_filename).strip()
            document_id = doc_key  # stable for chunk ids

            metadata: Dict[str, Any] = {
                "original_name": original_name,
                "filename": original_name or stored_filename,
                "stored_filename": stored_filename,
                "method": file_ext.replace(".", ""),
                "converted_format": "html",
                "doc_key": doc_key,
            }
            metadata = {k: v for k, v in metadata.items() if v is not None and str(v).strip()}

            combined_parts: List[str] = []
            if content:
                combined_parts.append(content)

            if tables_md:
                table_blocks: List[str] = []
                for idx, md in enumerate(tables_md, start=1):
                    table_blocks.append(f"جدول رقم {idx}:\n{md}")
                combined_parts.append("\n\n".join(table_blocks))

            raw_text_for_llm = "\n\n".join(combined_parts).strip()

            if not self.llm_service or not hasattr(self.llm_service, "to_structured_html"):
                raise RuntimeError(
                    "llm_service.to_structured_html() غير متوفر. "
                    "تأكد من تعديل llm_service.py وتمريره هنا."
                )

            # ✅ حماية نهائية فقط
            if len(raw_text_for_llm) > self.html_hard_max_chars:
                raw_text_for_llm = raw_text_for_llm[: self.html_hard_max_chars]
                print(f"✂️ Truncated raw text to {len(raw_text_for_llm)} chars (HTML_HARD_MAX_CHARS)")

            segments = self._split_text_segments(raw_text_for_llm)
            print(f"🧩 HTML convert segments={len(segments)} total_chars={len(raw_text_for_llm)} timeout={self.html_llm_timeout}s")

            # ✅ اجمع BODY فقط من كل segment (بدون doctype/html/head)
            body_parts: List[str] = []
            save_name = original_name or stored_filename

            for si, seg in enumerate(segments, start=1):
                seg = (seg or "").strip()
                if not seg:
                    continue
                print(f"🚀 Segment {si}/{len(segments)} chars={len(seg)}")

                try:
                    out = await asyncio.wait_for(
                        self._maybe_await(self.llm_service.to_structured_html(seg)),
                        timeout=self.html_llm_timeout,
                    )
                except asyncio.TimeoutError:
                    print(f"⏱️ Segment {si} timed out")
                    continue

                inner = self._strip_outer_html(out or "")
                if inner:
                    body_parts.append(inner)

            body_inner_all = "\n\n".join(body_parts).strip()
            if not body_inner_all:
                print("⚠️ LLM رجّع HTML فارغ")
                return {"html": "", "converted_txt_path": "", "converted_name": ""}

            # ✅ وثيقة HTML واحدة فقط
            html = self._wrap_html_document(body_inner_all, title=save_name)

            converted_txt_path, converted_name = self._save_converted_html_as_txt(save_name, html)

            # ✅ chunking من HTML واحد
            chunks_html = self._split_html(html)
            print(f"🧩 HTML chunks produced: {len(chunks_html)}")

            if not chunks_html:
                return {"html": html, "converted_txt_path": converted_txt_path, "converted_name": converted_name}

            self._delete_existing_index(doc_key)

            # ✅ IMPORTANT:
            # نفهرس "نص" عشان يجاوب حتى لو المعلومة في <p>
            # ونحفظ HTML للعرض داخل metadata
            if self.rag_system:
                for idx, chunk_html in enumerate(chunks_html, start=0):
                    chunk_html = (chunk_html or "").strip()
                    if not chunk_html:
                        continue

                    chunk_plain = self._html_to_plain(chunk_html)
                    if not chunk_plain:
                        continue

                    chunk_id = f"{document_id}::chunk_{idx}"

                    chunk_meta = dict(metadata)
                    chunk_meta["converted_file"] = converted_txt_path
                    chunk_meta["skip_chunking"] = True
                    chunk_meta["chunk_index"] = int(idx)

                    # ✅ store HTML for UI preview (optional)
                    chunk_meta["html"] = chunk_html

                    self.rag_system.add_document(
                        content=chunk_plain,   # ✅ indexed content
                        metadata=chunk_meta,
                        document_id=chunk_id,
                    )

            return {"html": html, "converted_txt_path": converted_txt_path, "converted_name": converted_name}

        except Exception as e:
            print(f"❌ DocumentProcessor error: {e}")
            raise

    # =========================================================
    # OCR deps (optional)
    # =========================================================
    def _try_import_ocr(self):
        if not self.enable_ocr:
            return None

        try:
            import pytesseract
            from PIL import Image
            from pdf2image import convert_from_path

            if self.tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

            return pytesseract, Image, convert_from_path
        except Exception as e:
            print("⚠️ OCR enabled لكن مكتبات OCR غير جاهزة:", e)
            return None

    # =========================================================
    # File handlers
    # =========================================================
    def _process_image(self, file_path: str) -> str:
        if not self.enable_ocr:
            return ""

        ocr = self._try_import_ocr()
        if not ocr:
            return ""

        pytesseract, Image, _convert_from_path = ocr
        text = pytesseract.image_to_string(Image.open(file_path), lang="ara+eng")
        return self._clean_text_keep_tables(text)

    def _process_pdf(self, file_path: str) -> str:
        reader = PdfReader(file_path)
        text_parts: List[str] = []

        for page in reader.pages:
            extracted = page.extract_text() or ""
            if extracted.strip():
                text_parts.append(extracted)

        text = "\n\n".join(text_parts).strip()
        if self.enable_ocr and len(text.strip()) < 20:
            ocr = self._try_import_ocr()
            if ocr:
                pytesseract, _Image, convert_from_path = ocr
                try:
                    images = convert_from_path(file_path)
                    ocr_parts: List[str] = []
                    for img in images:
                        ocr_parts.append(pytesseract.image_to_string(img, lang="ara+eng"))
                    text = "\n\n".join(ocr_parts).strip()
                except Exception as e:
                    print("⚠️ PDF OCR failed:", e)

        return self._clean_text_keep_tables(text)

    def _process_docx_with_tables(self, file_path: str) -> Tuple[str, List[str]]:
        doc = DocxDocument(file_path)

        paras: List[str] = []
        for p in doc.paragraphs:
            t = (p.text or "").strip()
            if t:
                paras.append(t)

        tables_md: List[str] = []
        for table in doc.tables:
            md = self._docx_table_to_markdown(table).strip()
            if md:
                tables_md.append(md)

        content = "\n".join(paras).strip()
        return self._clean_text_keep_tables(content), tables_md

    def _process_txt(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return self._clean_text_keep_tables(f.read())

    # =========================================================
    # DOCX table -> Markdown
    # =========================================================
    def _docx_table_to_markdown(self, table) -> str:
        rows: List[List[str]] = []
        max_cols = 0

        for row in table.rows:
            cells = []
            for cell in row.cells:
                txt = (cell.text or "").strip()
                txt = re.sub(r"\s+", " ", txt).strip()
                cells.append(txt)
            max_cols = max(max_cols, len(cells))
            rows.append(cells)

        if not rows or max_cols == 0:
            return ""

        norm_rows: List[List[str]] = []
        for r in rows:
            rr = r + [""] * (max_cols - len(r))
            norm_rows.append(rr)

        def esc(x: str) -> str:
            x = (x or "").strip()
            return x.replace("|", "\\|")

        def _norm_cell(s: str) -> str:
            s = (s or "").strip()
            s = re.sub(r"\s+", " ", s)
            return s

        def _looks_like_caption_row(row_vals: List[str]) -> bool:
            cleaned = [_norm_cell(c) for c in row_vals if _norm_cell(c)]
            if not cleaned:
                return False
            if len(set(cleaned)) == 1:
                return True
            if len(cleaned) == 1 and len(cleaned[0]) >= 10:
                return True
            return False

        caption = ""
        data_rows = norm_rows

        if _looks_like_caption_row(norm_rows[0]):
            caption = _norm_cell(norm_rows[0][0]) or next((c for c in norm_rows[0] if _norm_cell(c)), "")
            data_rows = norm_rows[1:] if len(norm_rows) > 1 else []

        if not data_rows:
            return f"**{caption}**" if caption else ""

        if max_cols == 2:
            header = ["المعدل", "التقدير"]
            body = data_rows
        else:
            header = data_rows[0]
            body = data_rows[1:] if len(data_rows) > 1 else []

        header_line = "| " + " | ".join(esc(x) for x in header) + " |"
        sep_line = "| " + " | ".join(["---"] * max_cols) + " |"

        body_lines = []
        for r in body:
            body_lines.append("| " + " | ".join(esc(x) for x in r) + " |")

        if caption:
            return ("**" + esc(caption) + "**\n\n" + "\n".join([header_line, sep_line] + body_lines)).strip()

        return "\n".join([header_line, sep_line] + body_lines).strip()

    # =========================================================
    # Cleaning
    # =========================================================
    def _clean_text_keep_tables(self, text: str) -> str:
        if not text:
            return ""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()