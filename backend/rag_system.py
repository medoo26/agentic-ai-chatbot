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
    def __init__(self):
        self.api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        self.vector_store_path = os.getenv("VECTOR_STORE_PATH", "./vector_store")

        # مهم: لا تستخدم نفس collection القديمة مع موديل جديد
        self.collection_name = os.getenv("CHROMA_COLLECTION", "university_docs_e5_v1")

        self.embedding_mode = (os.getenv("EMBEDDING_MODE", "local") or "local").strip().lower()
        if self.embedding_mode not in ("local", "openai"):
            self.embedding_mode = "local"

        self.force_e5 = (os.getenv("FORCE_E5", "0").strip() == "1")

        self.embed_mode: Optional[str] = None
        self.openai_embeddings = None
        self.local_model = None
        self.local_model_name: str = ""

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

        default_local = "intfloat/multilingual-e5-large"
        local_name = (os.getenv("LOCAL_EMBEDDING_MODEL") or "").strip()

        if self.force_e5 or not local_name:
            local_name = default_local

        try:
            self.local_model = SentenceTransformer(local_name)
            self.local_model_name = local_name
            self.embed_mode = "local"
            print(f"✅ Embeddings: LOCAL mode ({local_name})")
        except Exception as e:
            self.embed_mode = None
            print(f"❌ Local embeddings failed: {e}")

    # -----------------------------
    # E5 prefix helpers
    # -----------------------------
    def _is_e5(self) -> bool:
        name = (self.local_model_name or "").lower()
        return "e5" in name or "intfloat" in name

    def _embed_texts(self, texts: List[str]) -> Optional[List[List[float]]]:
        if not texts:
            return []
        if self.embed_mode == "openai" and self.openai_embeddings is not None:
            return self.openai_embeddings.embed_documents(texts)

        if self.embed_mode == "local" and self.local_model is not None:
            if self._is_e5():
                texts = [f"passage: {t}" for t in texts]
            vecs = self.local_model.encode(texts, normalize_embeddings=True)
            return [v.tolist() for v in vecs]

        return None

    def _embed_query(self, q: str) -> Optional[List[float]]:
        if not q:
            return None
        if self.embed_mode == "openai" and self.openai_embeddings is not None:
            return self.openai_embeddings.embed_query(q)

        if self.embed_mode == "local" and self.local_model is not None:
            if self._is_e5():
                q = f"query: {q}"
            v = self.local_model.encode([q], normalize_embeddings=True)[0]
            return v.tolist()

        return None

    # -----------------------------
    # delete helper (used by DocumentProcessor)
    # -----------------------------
    def delete_doc_key(self, doc_key: str) -> None:
        dk = (doc_key or "").strip()
        if not dk:
            return
        try:
            self.collection.delete(where={"doc_key": dk})
        except Exception as e:
            print("⚠️ delete_doc_key failed:", e)

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

                # normalize article_no + article_no_norm
                article_no = str(md.get("article_no") or "").strip()
                if article_no:
                    article_no = _normalize_digits(article_no)
                    md["article_no"] = article_no
                    md["article_no_norm"] = _normalize_ar(article_no)

                # استخراج article_no تلقائياً من النص إذا غير موجود
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
    # Neighbor helpers
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
                meta0 = metas[0] or {}
                page_number = meta0.get("page_number")
                try:
                    page_number = int(page_number) if page_number is not None else None
                except Exception:
                    page_number = None

                return {
                    "content": (docs[0] or "").strip(),
                    "metadata": meta0,
                    "score": None,
                    "page_number": page_number,
                }
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
                    meta0 = m or {}
                    page_number = meta0.get("page_number")
                    try:
                        page_number = int(page_number) if page_number is not None else None
                    except Exception:
                        page_number = None

                    return {
                        "content": (d or "").strip(),
                        "metadata": meta0,
                        "score": None,
                        "page_number": page_number,
                    }
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
    # Semantic search ONLY
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

        where_candidates: List[Optional[Dict[str, Any]]] = [where]

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

                meta = meta or {}
                page_number = meta.get("page_number")
                try:
                    page_number = int(page_number) if page_number is not None else None
                except Exception:
                    page_number = None

                out_local.append(
                    {
                        "content": txt,
                        "metadata": meta,
                        "score": float(dist) if dist is not None else None,
                        "matched_query": qq,
                        "page_number": page_number,
                    }
                )
            return out_local

        for w in where_candidates:
            for qq in variants:
                try:
                    collected.extend(_run_query(qq, w))
                except Exception as e:
                    print(f"❌ Search Error: {e}")

        if not collected:
            return []

        # de-dupe
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
# استخراج رقم المادة تلقائياً من نص الـ chunk (للـ metadata فقط)
# =========================================================
_RE_CHUNK_ARTICLE = re.compile(
    r"<h[234]>\s*(?:لل|ل|بال|وال|فال)?م[اآ]د[هة]\s+([^<\n]{1,60}?)\s*:?\s*</h[234]>"
    r"|<p>\s*(?:لل|ل|بال|وال|فال)?م[اآ]د[هة]\s+([^<\n]{1,60}?)\s*:?\s*</p>"
    r"|^(?:لل|ل|بال|وال|فال)?م[اآ]د[هة]\s+([^\n\.<،,:؛!\?؟]{1,60}?)(?:\s*:|(?=\s*\n)|$)",
    re.IGNORECASE | re.MULTILINE,
)


def _extract_article_no_from_chunk(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    m = _RE_CHUNK_ARTICLE.search(t)
    if not m:
        return ""
    val = (m.group(1) or m.group(2) or m.group(3) or "").strip()
    val = re.sub(r"\s+", " ", val).strip()
    val = val.replace(":", "").strip()
    val = re.split(r"\s+(?:من|في|عن|الفصل)\b", val)[0].strip()
    val = _normalize_digits(val)
    return val[:50] if val else ""