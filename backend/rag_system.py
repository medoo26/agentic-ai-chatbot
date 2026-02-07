# rag_system.py
import os
import re
import hashlib
from typing import List, Dict, Optional, Tuple, Any

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

# -----------------------------
# Helpers: Arabic normalization
# -----------------------------
_AR_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0653-\u065F]")
_AR_TATWEEL = re.compile(r"[ـ]+")

# ✅ Table blocks markers (يجي من DocumentProcessor)
_RE_TABLE_BLOCK = re.compile(r"\[\[TABLE\]\]\s*(.*?)\s*\[\[/TABLE\]\]", re.DOTALL | re.IGNORECASE)

# ✅ داخل بلوك الجدول نلقط TABLE_ID
_RE_TABLE_ID = re.compile(r"^\s*TABLE_ID\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)

# ✅ استخراج قسم markdown فقط من بلوك الجدول
_RE_MD_SECTION = re.compile(
    r"FORMAT:\s*markdown\s*(.*?)\s*(?:FORMAT:\s*plain|$)",
    re.DOTALL | re.IGNORECASE,
)


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


def _safe_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chroma metadata لازم تكون قيمها: str/int/float/bool
    وممنوع None / dict / list
    """
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


# ------------------------------------------
# Arabic-friendly chunking pre-processing
# ------------------------------------------
def _prepare_for_arabic_chunking(text: str) -> str:
    """
    تجهيز للنص العربي:
    - يحافظ على الفقرات
    - يضيف فواصل بعد نهايات الجمل
    - يفصل القوائم
    """
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
    """
    إذا لقينا فقرة قصيرة جدًا (عنوان)، ندمجها مع الفقرة اللي بعدها
    """
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
    """
    ✅ يرجّع Markdown Table فقط بدون:
    - [[TABLE]] / [[/TABLE]]
    - TABLE_ID / ORIGINAL_NAME
    - FORMAT: plain
    """
    t = (text or "").strip()
    if not t:
        return ""

    # شيل الغلاف
    t = re.sub(r"^\s*\[\[TABLE\]\]\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*\[\[/TABLE\]\]\s*$", "", t, flags=re.IGNORECASE)

    # خذ قسم markdown فقط
    m = _RE_MD_SECTION.search(t)
    if m:
        t = (m.group(1) or "").strip()

    if not t:
        return ""

    # خذ فقط سطور الجدول التي تحتوي |
    lines = [ln.rstrip() for ln in t.splitlines() if ln.strip()]
    table_lines = [ln for ln in lines if "|" in ln]

    if not table_lines:
        # fallback: نظّف أي meta lines
        t = re.sub(r"^\s*(TABLE_ID|ORIGINAL_NAME)\s*:\s*.*$", "", t, flags=re.IGNORECASE | re.MULTILINE)
        t = re.sub(r"^\s*FORMAT\s*:\s*(markdown|plain)\s*$", "", t, flags=re.IGNORECASE | re.MULTILINE)
        return t.strip()

    return "\n".join(table_lines).strip()


# ------------------------------------------
# Table-aware chunking (الجداول كقطعة وحدة)
# ------------------------------------------
def arabic_chunk(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    ✅ تقطيع عربي قوي + يدعم الجداول:
    - أي شيء داخل [[TABLE]] ... [[/TABLE]] يُحفظ كـ chunk واحد (is_table=True)
      (نخزن البلوك كامل عشان يبقى محمي من التشكنق)
    - نقرأ TABLE_ID ونخزّنه كـ table_key
    - باقي النص يتشانكنق عادي
    """
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
            table_chunks.append({
                "text": full_block,
                "is_table": True,
                "table_key": table_key,
            })
        return "\n\n"  # احذفها من النص الأساسي

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


# -----------------------------
# Query variants (typo/dialect tolerant)
# -----------------------------
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


# ==========================
# RAG System
# ==========================
class RAGSystem:
    def __init__(self):
        self.api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        self.vector_store_path = os.getenv("VECTOR_STORE_PATH", "./vector_store")
        self.collection_name = os.getenv("CHROMA_COLLECTION", "university_docs")

        # ✅ EMBEDDING_MODE: local | openai
        self.embedding_mode = (os.getenv("EMBEDDING_MODE", "local") or "local").strip().lower()
        if self.embedding_mode not in ("local", "openai"):
            self.embedding_mode = "local"

        self.embed_mode: Optional[str] = None  # "openai" | "local"
        self.openai_embeddings = None
        self.local_model = None

        self._init_embeddings()

        self.client = chromadb.PersistentClient(path=self.vector_store_path)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    # --------------------------
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
            self.openai_embeddings = OpenAIEmbeddings(
                model=emb_model,
                openai_api_key=self.api_key,
            )
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

    # --------------------------
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

    # --------------------------
    def add_document(self, content: str, metadata: Dict, document_id: str):
        self.add_documents([content], [metadata], document_id=document_id)

    def add_documents(self, texts: List[str], metadatas: List[Dict], document_id: Optional[str] = None):
        if self.embed_mode is None:
            print("❌ Cannot add documents: Embeddings not ready")
            return

        for i, text in enumerate(texts):
            raw = (text or "").strip()
            if not raw:
                continue

            md_in = metadatas[i] or {}
            md_in = dict(md_in)

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

            chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
            chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))

            chunks_obj = arabic_chunk(raw, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            if not chunks_obj:
                continue

            docs: List[str] = []
            ids: List[str] = []
            metadatas_list: List[Dict[str, Any]] = []

            base = _hash_id(doc_key)

            for j, obj in enumerate(chunks_obj):
                chunk_text = (obj.get("text") or "").strip()
                if not chunk_text:
                    continue

                is_table = bool(obj.get("is_table"))
                table_key = (obj.get("table_key") or "").strip()

                ids.append(f"{base}_{j}")
                docs.append(chunk_text)

                md = dict(meta)
                md["chunk_index"] = j
                md["chunk_len"] = len(chunk_text)
                md["is_table"] = bool(is_table)

                if is_table and table_key:
                    md["table_key"] = table_key

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

    # --------------------------
    def _fetch_doc_chunks(self, doc_key: str) -> List[Tuple[int, str, Dict[str, Any]]]:
        dk = (doc_key or "").strip()
        if not dk:
            return []
        try:
            res = self.collection.get(where={"doc_key": dk}, include=["documents", "metadatas"])
            docs = res.get("documents") or []
            metas = res.get("metadatas") or []
            items: List[Tuple[int, str, Dict[str, Any]]] = []
            for d, m in zip(docs, metas):
                if not d:
                    continue
                mi = m or {}
                idx = int(mi.get("chunk_index") or 0)
                items.append((idx, d, mi))
            items.sort(key=lambda x: x[0])
            return items
        except Exception:
            return []

    def _attach_neighbors(self, hit: Dict[str, Any], window: int = 1) -> List[Dict[str, Any]]:
        meta = hit.get("metadata") or {}
        doc_key = meta.get("doc_key")
        idx = meta.get("chunk_index")

        if doc_key is None or idx is None:
            return [hit]

        try:
            idx = int(idx)
        except Exception:
            return [hit]

        all_chunks = self._fetch_doc_chunks(str(doc_key))
        if not all_chunks:
            return [hit]

        mp = {i: (t, m) for i, t, m in all_chunks}
        start = max(0, idx - window)
        end = idx + window

        out: List[Dict[str, Any]] = []
        for j in range(start, end + 1):
            if j not in mp:
                continue
            t, m = mp[j]
            out.append({
                "content": t,
                "metadata": m or {},
                "score": hit.get("score"),
                "neighbor_of": idx,
            })
        return out or [hit]

    # ✅ جلب الجدول كامل بواسطة table_key فقط
    def _fetch_full_table(self, table_key: str) -> Optional[str]:
        tk = (table_key or "").strip()
        if not tk:
            return None

        try:
            res = self.collection.get(
                where={"is_table": True, "table_key": tk},
                include=["documents", "metadatas"],
            )
            docs = res.get("documents") or []
            if not docs:
                return None

            full = "\n\n".join([d for d in docs if d]).strip()
            if not full:
                return None

            md = _extract_markdown_from_table_block(full)
            return (md or "").strip() or None

        except Exception:
            return None

    # --------------------------
    def search(
        self,
        query: str,
        top_k: int = 6,
        include_neighbors: bool = True,
        neighbor_window: int = 1,
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

        collected: List[Dict[str, Any]] = []

        for qq in variants:
            try:
                q_emb = self._embed_query(qq)
                if q_emb is None:
                    continue

                results = self.collection.query(
                    query_embeddings=[q_emb],
                    n_results=top_k,
                    where=where,
                    include=["documents", "metadatas", "distances"],
                )

                docs = (results.get("documents") or [[]])[0]
                metas = (results.get("metadatas") or [[]])[0]
                dists = (results.get("distances") or [[]])[0]

                for doc, meta, dist in zip(docs, metas, dists):
                    collected.append({
                        "content": doc,
                        "metadata": meta or {},
                        "score": float(dist) if dist is not None else None,
                        "matched_query": qq,
                    })

            except Exception as e:
                print(f"❌ Search Error: {e}")

        if not collected:
            return []

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

        # ✅ إذا أفضل نتيجة جدول: رجّع الجدول كامل (Markdown فقط)
        if top_hits:
            best = top_hits[0]
            bm = best.get("metadata") or {}
            if bm.get("is_table") and bm.get("table_key"):
                table_key = str(bm.get("table_key") or "")
                full = self._fetch_full_table(table_key)
                if full:
                    best = dict(best)
                    best["content"] = full
                    best["is_full_table"] = True
                    top_hits[0] = best

        if not include_neighbors:
            return top_hits

        merged: List[Dict[str, Any]] = []
        for h in top_hits[: max(1, top_k // 2)]:
            if (h.get("metadata") or {}).get("is_table") or h.get("is_full_table"):
                merged.append(h)
                continue
            merged.extend(self._attach_neighbors(h, window=neighbor_window))

        seen2 = set()
        final: List[Dict[str, Any]] = []
        for it in merged:
            meta = it.get("metadata") or {}
            dk = str(meta.get("doc_key") or "")
            ci = str(meta.get("chunk_index") or "")
            hsh = hashlib.md5((it.get("content") or "").encode("utf-8", errors="ignore")).hexdigest()[:10]
            key = f"{dk}:{ci}:{hsh}"
            if key in seen2:
                continue
            seen2.add(key)
            final.append(it)

        def _sort_key(x):
            m = x.get("metadata") or {}
            return (str(m.get("doc_key") or ""), int(m.get("chunk_index") or 0))

        final.sort(key=_sort_key)

        limit = max(top_k, len(final))
        return final[:limit]
