# document_processor.py
import os
import re
from typing import Dict, List, Tuple, Optional

from pypdf import PdfReader
from docx import Document as DocxDocument


class DocumentProcessor:
    def __init__(self, rag_system):
        self.rag_system = rag_system

        # ✅ OCR اختياري (افتراضيًا OFF)
        # فعّلها بوضع: ENABLE_OCR=1 في .env
        self.enable_ocr = (os.getenv("ENABLE_OCR", "0").strip() == "1")

        # لو تحتاج مسار tesseract (اختياري)
        # مثال: TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
        self.tesseract_cmd = (os.getenv("TESSERACT_CMD") or "").strip()

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

    # =========================================================
    # Main processor
    # =========================================================
    async def process_file(self, file_path: str) -> str:
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
                return ""

            stored_filename = os.path.basename(file_path)
            original_name = self._clean_original_name(stored_filename)

            # ✅ مهم: وثّق ID ثابت لمنع تكرار الفهرسة بسبب uuid في اسم التخزين
            document_id = (original_name or stored_filename).strip()

            metadata: Dict[str, str] = {
                "original_name": original_name,
                "filename": original_name or stored_filename,
                "stored_filename": stored_filename,
                "method": file_ext.replace(".", ""),
                "category": "جامعة",
            }
            metadata = {k: v for k, v in metadata.items() if v is not None and str(v).strip()}

            combined_parts: List[str] = []
            if content:
                combined_parts.append(content)

            # ✅ كل الجداول داخل نفس الوثيقة لكن كل جدول له ID
            if tables_md:
                table_blocks: List[str] = []
                for idx, md in enumerate(tables_md, start=1):
                    table_id = f"{document_id}::table_{idx}"
                    plain = self._markdown_table_to_plain_text(md)

                    block = (
                        f"[[TABLE]]\n"
                        f"TABLE_ID: {table_id}\n"
                        f"ORIGINAL_NAME: {original_name}\n"
                        f"FORMAT: markdown\n\n"
                        f"{md}\n\n"
                        f"FORMAT: plain\n"
                        f"{plain}\n"
                        f"[[/TABLE]]"
                    )
                    table_blocks.append(block)

                combined_parts.append("\n\n".join(table_blocks))

            combined = "\n\n".join(combined_parts).strip()

            if self.rag_system:
                self.rag_system.add_document(
                    content=combined,
                    metadata=metadata,
                    document_id=document_id,
                )

            return combined

        except Exception as e:
            print(f"❌ DocumentProcessor error: {e}")
            raise

    # =========================================================
    # OCR deps (اختياري)
    # =========================================================
    def _try_import_ocr(self):
        """
        ✅ OCR optional: ما نستورد هالمكتبات إلا إذا ENABLE_OCR=1
        """
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
            # ما نعالج الصور بدون OCR
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

        # ✅ fallback OCR فقط لو ENABLE_OCR=1 والنص ضعيف
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
    # DOCX table -> Markdown (FIX merged header row)
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

        # نكمّل الناقص (بسبب merged cells)
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

        # ✅ إذا أول صف عنوان مكرر/مدمج: نخليه Caption
        if _looks_like_caption_row(norm_rows[0]):
            caption = _norm_cell(norm_rows[0][0]) or next(
                (c for c in norm_rows[0] if _norm_cell(c)), ""
            )
            data_rows = norm_rows[1:] if len(norm_rows) > 1 else []

        if not data_rows:
            return f"**{caption}**" if caption else ""

        # ✅ لو الجدول عمودين: هيدر ثابت
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
            return (
                ("**" + esc(caption) + "**\n\n" + "\n".join([header_line, sep_line] + body_lines))
            ).strip()

        return "\n".join([header_line, sep_line] + body_lines).strip()

    # =========================================================
    # Markdown table -> plain text (لتحسين البحث)
    # =========================================================
    def _markdown_table_to_plain_text(self, md: str) -> str:
        lines = [ln.strip() for ln in (md or "").splitlines() if ln.strip()]
        if len(lines) < 2:
            return (md or "").strip()

        # إذا فيه Caption **...** فوق الجدول، تجاهله
        if lines and lines[0].startswith("**") and lines[0].endswith("**"):
            lines = [ln for ln in lines if "|" in ln]
            if len(lines) < 2:
                return (md or "").strip()

        header = [c.strip() for c in lines[0].strip("|").split("|")]
        data_lines = lines[2:] if len(lines) >= 3 else []

        out_rows: List[str] = []
        for ln in data_lines:
            cells = [c.strip() for c in ln.strip("|").split("|")]

            if len(cells) < len(header):
                cells += [""] * (len(header) - len(cells))
            if len(cells) > len(header):
                cells = cells[: len(header)]

            pairs = []
            for h, v in zip(header, cells):
                h = h.strip() or "عمود"
                v = v.strip()
                pairs.append(f"{h}: {v}")
            out_rows.append(" | ".join(pairs))

        return "\n".join(out_rows).strip()

    # =========================================================
    # Cleaning (keep table formatting)
    # =========================================================
    def _clean_text_keep_tables(self, text: str) -> str:
        if not text:
            return ""

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()
