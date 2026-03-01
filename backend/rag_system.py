# rag_system.py
import os
import re
import hashlib
from typing import List, Dict, Optional, Any

import chromadb
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

# OpenAI embeddings (اختياري)
try:
    from langchain_openai import OpenAIEmbeddings
except Exception:
    OpenAIEmbeddings = None

# Local embeddings
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

load_dotenv()

_AR_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0653-\u065F]")
_AR_TATWEEL = re.compile(r"[ـ]+")

_RE_TABLE_BLOCK = re.compile(r"\[\[TABLE\]\]\s*(.*?)\s*\[\[/TABLE\]\]", re.DOTALL | re.IGNORECASE)
_RE_TABLE_ID = re.compile(r"^\s*TABLE_ID\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_RE_MD_SECTION = re.compile(
    r"FORMAT:\s*markdown\s*(.*?)\s*(?:FORMAT:\s*plain|$)",
    re.DOTALL | re.IGNORECASE,
)

# ✅ قاموس الأرقام العربية الترتيبية - يغطي كل أشكال الكتابة
ARABIC_ORDINALS_MAP: Dict[str, List[str]] = {
    "الأولى":                   ["1", "أولى", "اولى", "الاولى"],
    "الثانية":                  ["2", "ثانية", "ثانيه", "الثانيه"],
    "الثالثة":                  ["3", "ثالثة", "ثالثه", "الثالثه"],
    "الرابعة":                  ["4", "رابعة", "رابعه", "الرابعه"],
    "الخامسة":                  ["5", "خامسة", "خامسه", "الخامسه"],
    "السادسة":                  ["6", "سادسة", "سادسه", "السادسه"],
    "السابعة":                  ["7", "سابعة", "سابعه", "السابعه"],
    "الثامنة":                  ["8", "ثامنة", "ثامنه", "الثامنه"],
    "التاسعة":                  ["9", "تاسعة", "تاسعه", "التاسعه"],
    "العاشرة":                  ["10", "عاشرة", "عاشره", "العاشره"],
    "الحادية عشرة":             ["11", "الحادية عشر", "الحاديه عشره"],
    "الثانية عشرة":             ["12", "الثانية عشر", "الثانيه عشره"],
    "الثالثة عشرة":             ["13", "الثالثة عشر", "الثالثه عشره"],
    "الرابعة عشرة":             ["14", "الرابعة عشر", "الرابعه عشره"],
    "الخامسة عشرة":             ["15", "الخامسة عشر", "الخامسه عشره"],
    "السادسة عشرة":             ["16", "السادسة عشر", "السادسه عشره"],
    "السابعة عشرة":             ["17", "السابعة عشر", "السابعه عشره"],
    "الثامنة عشرة":             ["18", "الثامنة عشر", "الثامنه عشره"],
    "التاسعة عشرة":             ["19", "التاسعة عشر", "التاسعه عشره"],
    "العشرون":                  ["20", "العشرين"],
    "الحادية والعشرون":         ["21", "الحادية والعشرين", "الحاديه والعشرين"],
    "الثانية والعشرون":         ["22", "الثانية والعشرين", "الثانيه والعشرين"],
    "الثالثة والعشرون":         ["23", "الثالثة والعشرين", "الثالثه والعشرين"],
    "الرابعة والعشرون":         ["24", "الرابعة والعشرين", "الرابعه والعشرين"],
    "الخامسة والعشرون":         ["25", "الخامسة والعشرين", "الخامسه والعشرين"],
    "السادسة والعشرون":         ["26", "السادسة والعشرين", "السادسه والعشرين"],
    "السابعة والعشرون":         ["27", "السابعة والعشرين", "السابعه والعشرين"],
    "الثامنة والعشرون":         ["28", "الثامنة والعشرين", "الثامنه والعشرين"],
    "التاسعة والعشرون":         ["29", "التاسعة والعشرين", "التاسعه والعشرين"],
    "الثلاثون":                 ["30", "الثلاثين"],
    "الحادية والثلاثون":        ["31", "الحادية والثلاثين", "الحاديه والثلاثين"],
    "الثانية والثلاثون":        ["32", "الثانية والثلاثين", "الثانيه والثلاثين"],
    "الثالثة والثلاثون":        ["33", "الثالثة والثلاثين", "الثالثه والثلاثين"],
    "الرابعة والثلاثون":        ["34", "الرابعة والثلاثين", "الرابعه والثلاثين"],
    "الخامسة والثلاثون":        ["35", "الخامسة والثلاثين", "الخامسه والثلاثين"],
    "السادسة والثلاثون":        ["36", "السادسة والثلاثين", "السادسه والثلاثين"],
    "السابعة والثلاثون":        ["37", "السابعة والثلاثين", "السابعه والثلاثين"],
    "الثامنة والثلاثون":        ["38", "الثامنة والثلاثين", "الثامنه والثلاثين"],
    "التاسعة والثلاثون":        ["39", "التاسعة والثلاثين", "التاسعه والثلاثين"],
    "الأربعون":                 ["40", "الأربعين"],
    "الحادية والأربعون":        ["41", "الحادية والأربعين", "الحاديه والأربعين"],
    "الثانية والأربعون":        ["42", "الثانية والأربعين", "الثانيه والأربعين"],
    "الثالثة والأربعون":        ["43", "الثالثة والأربعين", "الثالثه والأربعين"],
    "الرابعة والأربعون":        ["44", "الرابعة والأربعين", "الرابعه والأربعين"],
    "الخامسة والأربعون":        ["45", "الخامسة والأربعين", "الخامسه والأربعين"],
    "السادسة والأربعون":        ["46", "السادسة والأربعين", "السادسه والأربعين"],
    "السابعة والأربعون":        ["47", "السابعة والأربعين", "السابعه والأربعين"],
    "الثامنة والأربعون":        ["48", "الثامنة والأربعين", "الثامنه والأربعين"],
    "التاسعة والأربعون":        ["49", "التاسعة والأربعين", "التاسعه والأربعين"],
    "الخمسون":                  ["50", "الخمسين"],
    "الحادية والخمسون":         ["51", "الحادية والخمسين", "الحاديه والخمسين"],
    "الثانية والخمسون":         ["52", "الثانية والخمسين", "الثانيه والخمسين"],
    "الثالثة والخمسون":         ["53", "الثالثة والخمسين", "الثالثه والخمسين"],
    "الرابعة والخمسون":         ["54", "الرابعة والخمسين", "الرابعه والخمسين"],
    "الخامسة والخمسون":         ["55", "الخامسة والخمسين", "الخامسه والخمسين"],
    "السادسة والخمسون":         ["56", "السادسة والخمسين", "السادسه والخمسين"],
    "السابعة والخمسون":         ["57", "السابعة والخمسين", "السابعه والخمسين"],
    "الثامنة والخمسون":         ["58", "الثامنة والخمسين", "الثامنه والخمسين"],
    "التاسعة والخمسون":         ["59", "التاسعة والخمسين", "التاسعه والخمسين"],
    "الستون":                   ["60", "الستين"],
    "الحادية والستون":          ["61", "الحادية والستين", "الحاديه والستين"],
    "الثانية والستون":          ["62", "الثانية والستين", "الثانيه والستين"],
    "الثالثة والستون":          ["63", "الثالثة والستين", "الثالثه والستين"],
    "الرابعة والستون":          ["64", "الرابعة والستين", "الرابعه والستين"],
    "الخامسة والستون":          ["65", "الخامسة والستين", "الخامسه والستين"],
    "السادسة والستون":          ["66", "السادسة والستين", "السادسه والستين"],
    "السابعة والستون":          ["67", "السابعة والستين", "السابعه والستين"],
    "الثامنة والستون":          ["68", "الثامنة والستين", "الثامنه والستين"],
    "التاسعة والستون":          ["69", "التاسعة والستين", "التاسعه والستين"],
    "السبعون":                  ["70", "السبعين"],
    "الحادية والسبعون":         ["71", "الحادية والسبعين"],
    "الثانية والسبعون":         ["72", "الثانية والسبعين"],
    "الثالثة والسبعون":         ["73", "الثالثة والسبعين"],
    "الرابعة والسبعون":         ["74", "الرابعة والسبعين"],
    "الخامسة والسبعون":         ["75", "الخامسة والسبعين"],
    "السادسة والسبعون":         ["76", "السادسة والسبعين"],
    "السابعة والسبعون":         ["77", "السابعة والسبعين"],
    "الثامنة والسبعون":         ["78", "الثامنة والسبعين"],
    "التاسعة والسبعون":         ["79", "التاسعة والسبعين"],
    "الثمانون":                 ["80", "الثمانين"],
    "الحادية والثمانون":        ["81", "الحادية والثمانين"],
    "الثانية والثمانون":        ["82", "الثانية والثمانين"],
    "الثالثة والثمانون":        ["83", "الثالثة والثمانين"],
    "الرابعة والثمانون":        ["84", "الرابعة والثمانين"],
    "الخامسة والثمانون":        ["85", "الخامسة والثمانين"],
    "السادسة والثمانون":        ["86", "السادسة والثمانين"],
    "السابعة والثمانون":        ["87", "السابعة والثمانين"],
    "الثامنة والثمانون":        ["88", "الثامنة والثمانين"],
    "التاسعة والثمانون":        ["89", "التاسعة والثمانين"],
    "التسعون":                  ["90", "التسعين"],
    "الحادية والتسعون":         ["91", "الحادية والتسعين"],
    "الثانية والتسعون":         ["92", "الثانية والتسعين"],
    "الثالثة والتسعون":         ["93", "الثالثة والتسعين"],
    "الرابعة والتسعون":         ["94", "الرابعة والتسعين"],
    "الخامسة والتسعون":         ["95", "الخامسة والتسعين"],
    "السادسة والتسعون":         ["96", "السادسة والتسعين"],
    "السابعة والتسعون":         ["97", "السابعة والتسعين"],
    "الثامنة والتسعون":         ["98", "الثامنة والتسعين"],
    "التاسعة والتسعون":         ["99", "التاسعة والتسعين"],
    "المئة":                    ["100", "المائة"],
}


def _normalize_ar(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    t = re.sub(_AR_TATWEEL, "", t)
    t = re.sub(_AR_DIACRITICS, "", t)
    t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    t = t.replace("ى", "ي")
    t = t.replace("ة", "ه")
    t = t.replace("ؤ", "و").replace("ئ", "ي")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _normalize_digits(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    return s.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")).strip()


def _safe_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    meta = meta or {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[str(k)] = v
        else:
            out[str(k)] = str(v)
    return out


def _hash_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:18]


def _basename(name: str) -> str:
    n = (name or "").strip()
    if not n:
        return ""
    n = n.replace("\\", "/")
    if "/" in n:
        n = n.split("/")[-1].strip()
    return n


def _prepare_for_arabic_chunking(text: str) -> str:
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"([\.!\?؟])\s+", r"\1\n", t)
    t = re.sub(r"(؛)\s+", r"\1\n", t)
    t = t.replace(" - ", "\n- ")
    t = t.replace("• ", "\n• ")
    t = re.sub(r"(\s)(\d+\s*[\)\-\.]\s+)", r"\n\2", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = "\n".join([ln.rstrip() for ln in t.split("\n")]).strip()
    return t


def _merge_headings_with_next(paragraphs: List[str]) -> List[str]:
    merged: List[str] = []
    i = 0
    while i < len(paragraphs):
        p = paragraphs[i].strip()
        if not p:
            i += 1
            continue
        is_heading_like = (len(p) <= 50 and ("\n" not in p) and not re.search(r"[\.!\?؟；؛]", p))
        if is_heading_like and i + 1 < len(paragraphs):
            nxt = paragraphs[i + 1].strip()
            if nxt:
                merged.append(p + "\n" + nxt)
                i += 2
                continue
        merged.append(p)
        i += 1
    return merged


def _extract_markdown_from_table_block(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    t = re.sub(r"^\s*\[\[TABLE\]\]\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*\[\[/TABLE\]\]\s*$", "", t, flags=re.IGNORECASE)
    m = _RE_MD_SECTION.search(t)
    if m:
        t = (m.group(1) or "").strip()
    if not t:
        return ""
    lines = [ln.rstrip() for ln in t.splitlines() if ln.strip()]
    table_lines = [ln for ln in lines if "|" in ln]
    if not table_lines:
        t = re.sub(r"^\s*(TABLE_ID|ORIGINAL_NAME)\s*:\s*.*$", "", t, flags=re.IGNORECASE | re.MULTILINE)
        t = re.sub(r"^\s*FORMAT\s*:\s*(markdown|plain)\s*$", "", t, flags=re.IGNORECASE | re.MULTILINE)
        return t.strip()
    return "\n".join(table_lines).strip()


def _get_article_variants(article_phrase: str) -> List[str]:
    """
    ✅ يرجع كل الأشكال الممكنة لرقم/اسم المادة.
    مثال: "الثانية" -> ["الثانية", "2", "ثانية", "ثانيه", "الثانيه", ...]
    مثال: "2"       -> ["2", "الثانية", "ثانية", ...]
    يشتغل على: أرقام هندية/عربية، كتابة بالتاء المربوطة/المفتوحة، بدون تشكيل
    """
    p = (article_phrase or "").strip()
    if not p:
        return []

    p_norm = _normalize_ar(p)
    p_digits = _normalize_digits(p)

    variants: set = set()
    variants.add(p)
    variants.add(p_norm)
    if p_digits:
        variants.add(p_digits)

    for key, vals in ARABIC_ORDINALS_MAP.items():
        key_norm = _normalize_ar(key)
        all_forms = [key, key_norm] + [_normalize_ar(v) for v in vals] + vals
        if p_norm in [_normalize_ar(f) for f in all_forms] or p_digits in vals or p in vals:
            variants.add(key)
            variants.add(key_norm)
            variants.update(vals)
            variants.update([_normalize_ar(v) for v in vals])
            break

    return [v for v in variants if v and v.strip()]


def arabic_chunk(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return []

    table_chunks: List[Dict[str, Any]] = []

    def _table_repl(m):
        full_block = (m.group(0) or "").strip()
        inner = (m.group(1) or "").strip()
        if inner:
            mm = _RE_TABLE_ID.search(inner)
            table_key = mm.group(1).strip() if mm else ""
            table_chunks.append({"text": full_block, "is_table": True, "table_key": table_key})
        return "\n\n"

    without_tables = re.sub(_RE_TABLE_BLOCK, _table_repl, raw).strip()

    normal_chunks: List[Dict[str, Any]] = []
    if without_tables:
        t = _prepare_for_arabic_chunking(without_tables)
        raw_paras = [p.strip() for p in re.split(r"\n\s*\n", t) if p.strip()]
        paras = _merge_headings_with_next(raw_paras)
        rebuilt = "\n\n".join(paras).strip()

        separators = ["\n\n", "\n", "• ", "- ", "؛", "؟", "!", ".", "…", "،", " ", ""]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )

        chunks = splitter.split_text(rebuilt)
        for c in chunks:
            cc = re.sub(r"\s+", " ", (c or "")).strip()
            if len(cc) >= 40:
                normal_chunks.append({"text": cc, "is_table": False})

    return normal_chunks + table_chunks


def _query_variants(q: str) -> List[str]:
    q0 = (q or "").strip()
    if not q0:
        return []
    q1 = _normalize_ar(q0)
    q2 = re.sub(r"[^\w\s\u0600-\u06FF]", " ", q0)
    q2 = re.sub(r"\s+", " ", q2).strip()
    q3 = _normalize_ar(q2)

    variants: List[str] = []
    for x in [q0, q1, q2, q3]:
        x = (x or "").strip()
        if x and x not in variants:
            variants.append(x)
    return variants


class RAGSystem:
    # ✅ detect exec rules request
    _RE_EXEC_RULES_Q = re.compile(r"\bالقواعد\s+التنفيذية\b", re.IGNORECASE)

    # ✅ يلتقط المادة بكل أشكالها: "المادة 4", "للمادة 4", "للمادة الخامسة", "بمادة 5"
    _RE_ARTICLE_PHRASE = re.compile(
        r"(?:لل|ل|بال|وال|فال)?م[اآ]د[هة]\s+([^\n\.<،,:؛!\?؟]{1,60})",
        re.IGNORECASE,
    )

    def __init__(self):
        self.api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        self.vector_store_path = os.getenv("VECTOR_STORE_PATH", "./vector_store")
        self.collection_name = os.getenv("CHROMA_COLLECTION", "university_docs")

        self.embedding_mode = (os.getenv("EMBEDDING_MODE", "local") or "local").strip().lower()
        if self.embedding_mode not in ("local", "openai"):
            self.embedding_mode = "local"

        self.embed_mode: Optional[str] = None
        self.openai_embeddings = None
        self.local_model = None

        self._init_embeddings()

        self.client = chromadb.PersistentClient(path=self.vector_store_path)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

        try:
            print("📦 Chroma collection count =", self.collection.count())
            print("📁 VECTOR_STORE_PATH =", self.vector_store_path)
            print("🧩 CHROMA_COLLECTION =", self.collection_name)
        except Exception as e:
            print("⚠️ count() failed:", e)

    def _init_embeddings(self):
        if self.embedding_mode == "local":
            self._init_local_embeddings()
            return
        if self.embedding_mode == "openai":
            ok = self._init_openai_embeddings()
            if ok:
                return
            self._init_local_embeddings()
            return

    def _init_openai_embeddings(self) -> bool:
        if not self.api_key or OpenAIEmbeddings is None:
            print("⚠️ OpenAI Embeddings not available -> fallback to LOCAL.")
            return False
        try:
            emb_model = (os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small").strip()
            self.openai_embeddings = OpenAIEmbeddings(model=emb_model, openai_api_key=self.api_key)
            _ = self.openai_embeddings.embed_query("اختبار")
            self.embed_mode = "openai"
            print(f"✅ Embeddings: OPENAI mode ({emb_model})")
            return True
        except Exception as e:
            print(f"⚠️ OpenAI Embeddings failed -> fallback to LOCAL. Reason: {e}")
            self.openai_embeddings = None
            return False

    def _init_local_embeddings(self):
        if SentenceTransformer is None:
            self.embed_mode = None
            print("❌ No embeddings available (sentence-transformers not installed).")
            return
        try:
            local_name = (os.getenv("LOCAL_EMBEDDING_MODEL") or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2").strip()
            self.local_model = SentenceTransformer(local_name)
            self.embed_mode = "local"
            print(f"✅ Embeddings: LOCAL mode ({local_name})")
        except Exception as e:
            self.embed_mode = None
            print(f"❌ Local embeddings failed: {e}")

    def _embed_texts(self, texts: List[str]) -> Optional[List[List[float]]]:
        if not texts:
            return []
        if self.embed_mode == "openai" and self.openai_embeddings is not None:
            return self.openai_embeddings.embed_documents(texts)
        if self.embed_mode == "local" and self.local_model is not None:
            vecs = self.local_model.encode(texts, normalize_embeddings=True)
            return [v.tolist() for v in vecs]
        return None

    def _embed_query(self, q: str) -> Optional[List[float]]:
        if not q:
            return None
        if self.embed_mode == "openai" and self.openai_embeddings is not None:
            return self.openai_embeddings.embed_query(q)
        if self.embed_mode == "local" and self.local_model is not None:
            v = self.local_model.encode([q], normalize_embeddings=True)[0]
            return v.tolist()
        return None

    def add_document(self, content: str, metadata: Dict, document_id: str):
        self.add_documents([content], [metadata], document_id=document_id)

    def add_documents(self, texts: List[str], metadatas: List[Dict], document_id: Optional[str] = None):
        if self.embed_mode is None:
            print("❌ Cannot add documents: Embeddings not ready")
            return
        if not texts:
            return

        for i, text in enumerate(texts):
            raw = (text or "").strip()
            if not raw:
                continue

            md_in = dict(metadatas[i] or {})
            skip_chunking = bool(md_in.get("skip_chunking") or md_in.get("prechunked"))

            original_name = _basename(str(md_in.get("original_name") or md_in.get("filename") or md_in.get("name") or ""))
            if not original_name and document_id:
                original_name = _basename(str(document_id))

            doc_key = _basename(
                str(
                    original_name
                    or md_in.get("doc_key")
                    or md_in.get("filename")
                    or md_in.get("original_name")
                    or document_id
                    or f"doc_{i}"
                )
            ).strip()

            meta = _safe_meta(md_in)
            if original_name:
                meta["original_name"] = str(original_name)
                meta["filename"] = str(original_name)
            meta["doc_key"] = doc_key

            base_source = str(document_id or doc_key or f"doc_{i}")
            base = _hash_id(base_source)

            chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
            chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))

            if skip_chunking:
                provided_idx = meta.get("chunk_index")
                try:
                    provided_idx_int = int(provided_idx) if provided_idx is not None else i
                except Exception:
                    provided_idx_int = i
                chunks_obj = [{"text": raw, "is_table": False, "provided_index": provided_idx_int}]
            else:
                chunks_obj = arabic_chunk(raw, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            if not chunks_obj:
                continue

            docs: List[str] = []
            ids: List[str] = []
            metadatas_list: List[Dict[str, Any]] = []

            for j, obj in enumerate(chunks_obj):
                chunk_text = (obj.get("text") or "").strip()
                if not chunk_text:
                    continue

                is_table = bool(obj.get("is_table"))
                table_key = (obj.get("table_key") or "").strip()
                jj = int(obj.get("provided_index", j)) if skip_chunking else j

                ids.append(f"{base}_{jj}")
                docs.append(chunk_text)

                md = dict(meta)
                md["chunk_index"] = int(jj)
                md["chunk_len"] = len(chunk_text)
                md["is_table"] = bool(is_table)
                if is_table and table_key:
                    md["table_key"] = table_key

                # ✅ normalize article_no + article_no_norm
                article_no = str(md.get("article_no") or "").strip()
                if article_no:
                    article_no = _normalize_digits(article_no)
                    md["article_no"] = article_no
                    md["article_no_norm"] = _normalize_ar(article_no)

                # ✅ استخراج article_no تلقائياً من نص الـ chunk إذا مو موجود في metadata
                if not md.get("article_no"):
                    auto_article = _extract_article_no_from_chunk(chunk_text)
                    if auto_article:
                        md["article_no"] = auto_article
                        md["article_no_norm"] = _normalize_ar(auto_article)
                        md["section_type"] = md.get("section_type") or "article"

                metadatas_list.append(_safe_meta(md))

            if not docs:
                continue

            embs = self._embed_texts(docs)
            if embs is None:
                print("❌ Embedding failed; skipping indexing.")
                continue

            try:
                self.collection.delete(ids=ids)
            except Exception:
                pass

            self.collection.add(
                documents=docs,
                ids=ids,
                embeddings=embs,
                metadatas=metadatas_list,
            )

            print(f"✅ Indexed/Upserted {len(docs)} chunks for doc_key={doc_key} (mode={self.embed_mode})")

    # =========================================================
    # ✅ Smart query helpers
    # =========================================================
    def _is_exec_rules_query(self, q: str) -> bool:
        return bool(self._RE_EXEC_RULES_Q.search((q or "").strip()))

    def _extract_article_no_from_query(self, q: str) -> str:
        """
        يستخرج رقم/اسم المادة من سؤال المستخدم
        مثال: "المادة الثانية من الفصل الأول" -> "الثانية"
        """
        t = (q or "").strip()
        if not t:
            return ""
        m = self._RE_ARTICLE_PHRASE.search(t)
        if not m:
            return ""
        val = (m.group(1) or "").strip()
        # قطع عند "من" أو "في" لو موجود
        val = re.split(r"\s+(?:من|في|عن|بـ|الفصل)\b", val)[0].strip()
        val = re.sub(r"\s+", " ", val).strip()
        val = val.replace(":", "").strip()
        val = _normalize_digits(val)
        return val[:50]

    def _merge_where(self, a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        a = a or None
        b = b or None
        if not a and not b:
            return None
        if a and not b:
            return a
        if b and not a:
            return b
        return {"$and": [a, b]}

    def _build_smart_where(self, query: str) -> Optional[Dict[str, Any]]:
        """
        ✅ محسّن: يبني where filter يغطي كل أشكال رقم المادة العربية
        يستخدم _get_article_variants للتطابق الدقيق
        """
        q = (query or "").strip()
        if not q:
            return None

        article_no = self._extract_article_no_from_query(q)
        wants_rules = self._is_exec_rules_query(q)

        smart: List[Dict[str, Any]] = []

        if article_no:
            variants = _get_article_variants(article_no)
            if variants:
                or_conditions: List[Dict[str, Any]] = []
                for v in variants:
                    v = (v or "").strip()
                    if v:
                        or_conditions.append({"article_no": v})
                        v_norm = _normalize_ar(v)
                        if v_norm and v_norm != v:
                            or_conditions.append({"article_no_norm": v_norm})

                if or_conditions:
                    smart.append({"$or": or_conditions})

        if wants_rules and article_no:
            smart.append({"section_type": "exec_rules"})
        elif article_no:
            # لا نقيّد بـ section_type لأن بعض الملفات ما عندها هذا الحقل
            pass

        if not smart:
            return None
        if len(smart) == 1:
            return smart[0]
        return {"$and": smart}

    # =========================================================
    # ✅ Neighbor helpers
    # =========================================================
    def _chroma_get_by_doc_and_chunk(self, doc_key: str, chunk_index: int) -> Optional[Dict[str, Any]]:
        try:
            res = self.collection.get(
                where={"$and": [{"doc_key": str(doc_key)}, {"chunk_index": int(chunk_index)}]},
                include=["documents", "metadatas"],
            )
            docs = res.get("documents") or []
            metas = res.get("metadatas") or []
            if docs and metas:
                return {"content": (docs[0] or "").strip(), "metadata": metas[0] or {}, "score": None}
        except Exception:
            pass

        try:
            res = self.collection.get(
                where={"doc_key": str(doc_key)},
                include=["documents", "metadatas"],
            )
            docs = res.get("documents") or []
            metas = res.get("metadatas") or []
            for d, m in zip(docs, metas):
                if (m or {}).get("chunk_index") == int(chunk_index):
                    return {"content": (d or "").strip(), "metadata": m or {}, "score": None}
        except Exception:
            pass

        return None

    def _attach_neighbors(self, hit: Dict[str, Any], window: int = 2) -> List[Dict[str, Any]]:
        meta = hit.get("metadata") or {}
        doc_key = str(meta.get("doc_key") or "").strip()
        if not doc_key:
            return [hit]

        try:
            center = int(meta.get("chunk_index") or 0)
        except Exception:
            center = 0

        base_score = hit.get("score")
        try:
            base_score_f = float(base_score) if base_score is not None else None
        except Exception:
            base_score_f = None

        out: List[Dict[str, Any]] = []
        for idx in range(center - window, center + window + 1):
            if idx < 0:
                continue
            if idx == center:
                out.append(hit)
                continue
            nb = self._chroma_get_by_doc_and_chunk(doc_key, idx)
            if nb and (nb.get("content") or "").strip():
                if base_score_f is not None:
                    nb["score"] = base_score_f + (0.001 * abs(idx - center))
                out.append(nb)

        def _k(x):
            mm = x.get("metadata") or {}
            try:
                return int(mm.get("chunk_index") or 0)
            except Exception:
                return 0

        out.sort(key=_k)
        return out

    # =========================================================
    # ✅ search_article_exact — بحث مباشر بالمادة (الحل الجذري)
    # =========================================================
    def search_article_exact(
        self,
        article_phrase: str,
        doc_key: Optional[str] = None,
        top_k: int = 5,
        include_neighbors: bool = True,
        neighbor_window: int = 1,
    ) -> List[Dict]:
        """
        ✅ بحث مباشر في الـ metadata بدون semantic search.
        يضمن إرجاع المادة الصحيحة حتى لو الـ embeddings ضعيفة على الأرقام.
        يشتغل على كل الملفات المرفوعة بدون أي تعديل إضافي.

        استخدمه كـ primary search لما يطلب المستخدم مادة محددة.
        """
        variants = _get_article_variants(article_phrase)
        if not variants:
            return []

        # بناء or_conditions
        or_conditions: List[Dict[str, Any]] = []
        for v in variants:
            v = (v or "").strip()
            if v:
                or_conditions.append({"article_no": v})
                v_norm = _normalize_ar(v)
                if v_norm and v_norm != v:
                    or_conditions.append({"article_no_norm": v_norm})

        if not or_conditions:
            return []

        where: Dict[str, Any] = {"$or": or_conditions}
        if doc_key:
            where = {"$and": [{"doc_key": str(doc_key)}, where]}

        try:
            res = self.collection.get(
                where=where,
                include=["documents", "metadatas"],
            )
            docs = res.get("documents") or []
            metas = res.get("metadatas") or []

            results: List[Dict[str, Any]] = []
            for d, m in zip(docs, metas):
                txt = (d or "").strip()
                if txt:
                    results.append({
                        "content": txt,
                        "metadata": m or {},
                        "score": 0.0,
                        "matched_query": f"exact:{article_phrase}",
                    })

            results.sort(key=lambda x: int((x.get("metadata") or {}).get("chunk_index") or 0))
            results = results[:top_k]

            if not results or not include_neighbors:
                return results

            # أضف neighbors لأغنى نتيجة
            expanded: List[Dict[str, Any]] = []
            for h in results:
                meta_h = h.get("metadata") or {}
                if meta_h.get("is_table"):
                    expanded.append(h)
                else:
                    expanded.extend(self._attach_neighbors(h, window=neighbor_window))

            seen = set()
            final: List[Dict[str, Any]] = []
            for it in expanded:
                meta_it = it.get("metadata") or {}
                key = f"{meta_it.get('doc_key')}:{meta_it.get('chunk_index')}"
                if key not in seen:
                    seen.add(key)
                    final.append(it)

            final.sort(key=lambda x: int((x.get("metadata") or {}).get("chunk_index") or 0))
            return final

        except Exception as e:
            print(f"❌ search_article_exact error: {e}")
            return []

    # =========================================================
    # ✅ search_smart — الدالة الرئيسية الموحدة (استخدمها بدل search)
    # =========================================================
    def search_smart(
        self,
        query: str,
        top_k: int = 10,
        doc_key: Optional[str] = None,
        include_neighbors: bool = True,
        neighbor_window: int = 2,
    ) -> List[Dict]:
        """
        ✅ الدالة الموحدة للبحث:
        1) إذا طُلبت مادة محددة -> exact match أولاً
        2) إذا ما رجع شيء -> semantic search
        3) إذا ما طُلبت مادة -> semantic search مباشرة
        """
        article_no = self._extract_article_no_from_query(query)

        if article_no:
            print(f"📌 Article phrase detected: {article_no} -> trying exact match first")
            exact_results = self.search_article_exact(
                article_phrase=article_no,
                doc_key=doc_key,
                top_k=top_k,
                include_neighbors=include_neighbors,
                neighbor_window=neighbor_window,
            )
            if exact_results:
                print(f"✅ Exact match found: {len(exact_results)} chunks")
                return exact_results
            print("⚠️ Exact match returned nothing -> fallback to semantic search")

        # semantic search
        where: Optional[Dict[str, Any]] = None
        if doc_key:
            where = {"doc_key": str(doc_key)}

        return self.search(
            query=query,
            top_k=top_k,
            include_neighbors=include_neighbors,
            neighbor_window=neighbor_window,
            where=where,
        )

    # =========================================================
    # ✅ Search الأصلية (semantic) — محسّنة
    # =========================================================
    def search(
        self,
        query: str,
        top_k: int = 10,
        include_neighbors: bool = True,
        neighbor_window: int = 2,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        if self.embed_mode is None:
            return []

        q = (query or "").strip()
        if not q:
            return []

        variants = _query_variants(q)
        if not variants:
            return []

        smart_where = self._build_smart_where(q)
        effective_where = self._merge_where(where, smart_where)

        where_candidates: List[Optional[Dict[str, Any]]] = []
        where_candidates.append(effective_where)
        if where is not None and where != effective_where:
            where_candidates.append(where)
        where_candidates.append(None)

        collected: List[Dict[str, Any]] = []

        def _run_query(qq: str, w: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
            out_local: List[Dict[str, Any]] = []
            q_emb = self._embed_query(qq)
            if q_emb is None:
                return out_local

            results = self.collection.query(
                query_embeddings=[q_emb],
                n_results=top_k,
                where=w,
                include=["documents", "metadatas", "distances"],
            )

            docs = (results.get("documents") or [[]])[0]
            metas = (results.get("metadatas") or [[]])[0]
            dists = (results.get("distances") or [[]])[0]

            for doc, meta, dist in zip(docs, metas, dists):
                txt = (doc or "").strip()
                if not txt:
                    continue
                out_local.append({
                    "content": txt,
                    "metadata": meta or {},
                    "score": float(dist) if dist is not None else None,
                    "matched_query": qq,
                })
            return out_local

        got_any = False
        for w in where_candidates:
            local_collected: List[Dict[str, Any]] = []
            for qq in variants:
                try:
                    local_collected.extend(_run_query(qq, w))
                except Exception as e:
                    print(f"❌ Search Error: {e}")

            if local_collected:
                collected.extend(local_collected)
                got_any = True
                if len(collected) >= (top_k * 2):
                    break

        if not got_any or not collected:
            return []

        # unique
        seen = set()
        uniq: List[Dict[str, Any]] = []
        for it in collected:
            meta = it.get("metadata") or {}
            dk = str(meta.get("doc_key") or "")
            ci = str(meta.get("chunk_index") or "")
            h = hashlib.md5((it.get("content") or "").encode("utf-8", errors="ignore")).hexdigest()[:10]
            key = f"{dk}:{ci}:{h}"
            if key in seen:
                continue
            seen.add(key)
            uniq.append(it)

        uniq.sort(key=lambda x: (x.get("score") is None, x.get("score", 1e9)))
        top_hits = uniq[:top_k]

        if not include_neighbors:
            return top_hits

        anchors = top_hits[: max(5, min(len(top_hits), top_k))]
        expanded: List[Dict[str, Any]] = []
        for h in anchors:
            meta = h.get("metadata") or {}
            if meta.get("is_table") or h.get("is_full_table"):
                expanded.append(h)
                continue
            expanded.extend(self._attach_neighbors(h, window=neighbor_window))

        seen2 = set()
        final: List[Dict[str, Any]] = []
        for it in expanded:
            meta = it.get("metadata") or {}
            dk = str(meta.get("doc_key") or "")
            ci = str(meta.get("chunk_index") or "")
            h = hashlib.md5((it.get("content") or "").encode("utf-8", errors="ignore")).hexdigest()[:10]
            key = f"{dk}:{ci}:{h}"
            if key in seen2:
                continue
            seen2.add(key)
            final.append(it)

        def _sort_key(x):
            m = x.get("metadata") or {}
            dk = str(m.get("doc_key") or "")
            try:
                ci = int(m.get("chunk_index") or 0)
            except Exception:
                ci = 0
            sc = x.get("score")
            sc2 = sc if isinstance(sc, (int, float)) else 1e9
            return (sc2, dk, ci)

        final.sort(key=_sort_key)
        return final


# =========================================================
# ✅ دالة مساعدة: استخراج رقم المادة تلقائياً من نص الـ chunk
# تُستخدم أثناء الـ indexing لحفظ article_no في الـ metadata
# =========================================================
_RE_CHUNK_ARTICLE = re.compile(
    # ✅ Pattern 1: داخل h2/h3/h4 (الأفضل)
    r"<h[234]>\s*(?:لل|ل|بال|وال|فال)?م[اآ]د[هة]\s+([^<\n]{1,60}?)\s*:?\s*</h[234]>"
    # ✅ Pattern 2: داخل <p> — مثل <p>المادة الثامنة والخمسون</p>
    r"|<p>\s*(?:لل|ل|بال|وال|فال)?م[اآ]د[هة]\s+([^<\n]{1,60}?)\s*:?\s*</p>"
    # ✅ Pattern 3: نص عادي في أول السطر
    r"|^(?:لل|ل|بال|وال|فال)?م[اآ]د[هة]\s+([^\n\.<،,:؛!\?؟]{1,60}?)(?:\s*:|(?=\s*\n)|$)",
    re.IGNORECASE | re.MULTILINE,
)


def _extract_article_no_from_chunk(text: str) -> str:
    """
    يستخرج رقم المادة من نص الـ chunk أثناء الـ indexing.
    يدعم:
    - <h3>المادة الثانية</h3>
    - <p>المادة الثامنة والخمسون</p>
    - نص عادي: "المادة الستون"
    - مع حروف جر: "للمادة 4"
    """
    t = (text or "").strip()
    if not t:
        return ""
    m = _RE_CHUNK_ARTICLE.search(t)
    if not m:
        return ""
    # يأخذ أول group غير فارغ من الثلاث patterns
    val = (m.group(1) or m.group(2) or m.group(3) or "").strip()
    val = re.sub(r"\s+", " ", val).strip()
    val = val.replace(":", "").strip()
    # قطع عند "من" أو "في" أو "الفصل"
    val = re.split(r"\s+(?:من|في|عن|الفصل)\b", val)[0].strip()
    val = _normalize_digits(val)
    return val[:50] if val else ""