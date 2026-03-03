# main.py
from __future__ import annotations

import os
import re
import uuid
import shutil
from datetime import datetime
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import or_, text

from admin_api import router as admin_router
from database import init_db, get_db, AdminUser, Document

# ✅ فقط RAGSystem (بدون search_smart / search_article_exact / _get_article_variants)
from rag_system import RAGSystem
from llm_service import LLMService
from document_processor import DocumentProcessor

load_dotenv()

app = FastAPI(title="PSU Chatbot Backend API", version="1.5.0")

# =========================================================
# Simple topic memory (Follow-up handling)
# =========================================================
LAST_TOPIC = ""

# ✅ تخزين خيارات الملفات المؤقتة لما نسأل المستخدم "أي ملف تقصد؟"
# ✅ أضفنا pending_user_text عشان لما يختار ملف (زر) نرجع للسؤال الأصلي ونجيب الجواب فوراً
PENDING_CHOICES: Dict[str, Any] = {
    "article_phrase": "",
    "candidates": [],
    "forced_doc_key": "",
    "pending_user_text": "",
    "ts": None,
}

_PRONOUNY = re.compile(
    r"(اهدافها|مهامها|شروطها|مواعيدها|رسومها|آليتها|طريقتها|كيفها|وش هي|وشو|هذي|هذا|تلك|ذي)\b"
)

app.include_router(admin_router)

cors_origins = os.getenv(
    "CORS_ORIGINS", "http://localhost:5173,http://localhost:3000"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in cors_origins if o.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# Services
rag_system = RAGSystem()
llm_service = LLMService()
doc_processor = DocumentProcessor(rag_system, llm_service=llm_service)


# =========================================================
# DB init + Default Admin
# =========================================================
def create_default_admin():
    db = next(get_db())
    admin_username = os.getenv("ADMIN_USERNAME", "admin").strip()
    admin_password = os.getenv("ADMIN_PASSWORD", "admin123").strip()
    try:
        admin = db.query(AdminUser).filter(AdminUser.username == admin_username).first()
        if not admin:
            hashed_password = pwd_context.hash(admin_password)
            admin = AdminUser(username=admin_username, password_hash=hashed_password)
            db.add(admin)
            db.commit()
            print(f"✅ Default admin created: {admin_username}")
    finally:
        db.close()


init_db()
create_default_admin()


# =========================================================
# Schemas
# =========================================================
class MessageCreate(BaseModel):
    content: str
    # ✅ من الفرونت وقت ضغط زر اختيار الملف
    choice_doc_key: Optional[str] = None


class AttachmentOut(BaseModel):
    name: str
    url: str
    mime: Optional[str] = None


class ChoiceOut(BaseModel):
    label: str
    doc_key: str


class MessageResponse(BaseModel):
    id: int
    content: str
    sender: str
    timestamp: datetime
    sources: List[str] = []
    attachments: List[AttachmentOut] = []
    # ✅ خيارات تظهر كأزرار في الواجهة
    choices: List[ChoiceOut] = []


# =========================================================
# Helpers
# =========================================================
def _guess_mime(filename: str) -> str:
    fn = (filename or "").lower()
    if fn.endswith(".pdf"):
        return "application/pdf"
    if fn.endswith(".docx"):
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if fn.endswith(".doc"):
        return "application/msword"
    if fn.endswith(".xlsx"):
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if fn.endswith(".xls"):
        return "application/vnd.ms-excel"
    if fn.endswith(".pptx"):
        return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    if fn.endswith(".ppt"):
        return "application/vnd.ms-powerpoint"
    if fn.endswith(".txt"):
        return "text/plain"
    return "application/octet-stream"


_AR_STOP = {
    "من",
    "في",
    "على",
    "الى",
    "إلى",
    "عن",
    "مع",
    "و",
    "او",
    "أو",
    "هذا",
    "هذه",
    "عطني",
    "ابغى",
    "أبغى",
    "ارسل",
    "أرسل",
    "نموذج",
    "ملف",
    "تحميل",
    "رابط",
    "لو",
    "ليه",
    "وش",
    "ايش",
    "وشو",
    "كيف",
    "ممكن",
    "فضلا",
    "فضلاً",
}


def _normalize_ar(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    s = s.replace("ة", "ه").replace("ى", "ي")
    trans = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    s = s.translate(trans)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# =========================================================
# ✅ Arabic text helpers
# =========================================================
_RE_ARTICLE_PHRASE = re.compile(
    r"(?:لل|ل|بال|وال|فال)?م[اآ]د[هة]\s+([^\n\.<،,:؛!\?؟]{1,60})",
    re.IGNORECASE,
)
_RE_EXEC_RULES = re.compile(r"\bالقواعد\s+التنفيذية\b", re.IGNORECASE)


def _is_exec_rules_query(user_text: str) -> bool:
    return bool(_RE_EXEC_RULES.search((user_text or "").strip()))


def _fix_ar_spacing(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    t = re.sub(r"(\S)(من)\b", r"\1 \2", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _extract_article_phrase(user_text: str) -> str:
    t = _fix_ar_spacing(user_text)
    m = _RE_ARTICLE_PHRASE.search(t)
    if not m:
        return ""
    phrase = f"المادة {m.group(1)}"
    phrase = re.split(r"\s+(?:من|في|عن|الفصل)\b", phrase)[0].strip()
    phrase = re.sub(r"\s+", " ", phrase).strip()
    return phrase[:60].strip()


def _extract_keywords(query: str) -> List[str]:
    q = _normalize_ar(query)
    tokens = re.findall(r"[a-z0-9]+|[\u0600-\u06FF]+", q)
    tokens = [t for t in tokens if len(t) >= 2 and t not in _AR_STOP]
    return tokens[:6]


def _find_best_document(db: Session, query: str) -> Optional[Document]:
    q = (query or "").strip()
    if not q:
        return None
    q_norm = _normalize_ar(q)
    if any(
        ext in q_norm
        for ext in [".pdf", ".docx", ".doc", ".txt", ".xlsx", ".pptx", ".ppt"]
    ):
        doc = (
            db.query(Document)
            .filter(Document.name.ilike(f"%{q}%"))
            .order_by(Document.upload_date.desc())
            .first()
        )
        if doc:
            return doc

    keys = _extract_keywords(q)
    if not keys:
        short = q[:20]
        return (
            db.query(Document)
            .filter(Document.name.ilike(f"%{short}%"))
            .order_by(Document.upload_date.desc())
            .first()
        )

    conditions = [Document.name.ilike(f"%{k}%") for k in keys]
    return (
        db.query(Document)
        .filter(or_(*conditions))
        .order_by(Document.upload_date.desc())
        .first()
    )


# =========================================================
# ✅ Auto-select best doc_key from hits
# =========================================================
def _pick_best_doc_key_from_hits(hits: List[Dict[str, Any]]) -> str:
    if not hits:
        return ""
    per_doc: Dict[str, List[float]] = {}
    for h in hits:
        meta = h.get("metadata") or {}
        dk = str(meta.get("doc_key") or "").strip()
        if not dk:
            continue
        sc = h.get("score")
        if sc is None:
            continue
        try:
            scf = float(sc)
        except Exception:
            continue
        per_doc.setdefault(dk, []).append(scf)

    if not per_doc:
        for h in hits:
            meta = h.get("metadata") or {}
            dk = str(meta.get("doc_key") or "").strip()
            if dk:
                return dk
        return ""

    best_dk = ""
    best_val = 1e9
    for dk, scores in per_doc.items():
        scores_sorted = sorted(scores)[:3]
        avg = sum(scores_sorted) / max(1, len(scores_sorted))
        if avg < best_val:
            best_val = avg
            best_dk = dk
    return best_dk


# =========================================================
# ✅ Article token helpers (semantic-only filtering)
# =========================================================
def _article_tokens(article_phrase: str) -> List[str]:
    ap = _fix_ar_spacing(article_phrase)
    ap = _normalize_ar(ap)
    ap = ap.replace("الماده ", "").replace("المادة ", "").strip()
    toks = re.findall(r"[0-9]+|[\u0600-\u06FF]+", ap)
    toks = [t.strip() for t in toks if t.strip()]
    STOP = {"من", "في", "على", "الى", "إلى", "عن", "مع", "و", "او", "أو"}
    toks = [t for t in toks if t not in STOP]
    return toks[:6]


def _filter_hits_for_article(
    hits: List[Dict[str, Any]], article_phrase: str
) -> List[Dict[str, Any]]:
    if not hits:
        return []
    toks = _article_tokens(article_phrase)
    if not toks:
        return []

    out: List[Dict[str, Any]] = []
    for h in hits:
        meta = h.get("metadata") or {}
        blob = " ".join(
            [
                _normalize_ar(str(meta.get("h2") or "")),
                _normalize_ar(str(meta.get("h3") or "")),
                _normalize_ar(str(h.get("content") or "")),
            ]
        ).strip()

        ok = True
        for t in toks:
            if t not in blob:
                ok = False
                break
        if ok:
            out.append(h)

    out.sort(
        key=lambda x: float(x.get("score", 1e9))
        if isinstance(x.get("score"), (int, float))
        else 1e9
    )
    return out


def _rank_doc_candidates_from_hits(
    hits: List[Dict[str, Any]], top_n: int = 6
) -> List[Dict[str, Any]]:
    best_per_doc: Dict[str, Dict[str, Any]] = {}
    for h in hits or []:
        meta = h.get("metadata") or {}
        dk = str(meta.get("doc_key") or "").strip()
        if not dk:
            continue
        sc = h.get("score")
        if not isinstance(sc, (int, float)):
            continue
        cur = best_per_doc.get(dk)
        if cur is None or float(sc) < float(cur["best_score"]):
            best_per_doc[dk] = {
                "doc_key": dk,
                "best_score": float(sc),
                "original_name": (meta.get("original_name") or meta.get("filename") or dk),
            }
    cands = list(best_per_doc.values())
    cands.sort(key=lambda x: x["best_score"])
    return cands[: max(1, top_n)]


def _pick_candidate_by_name(
    user_text: str, candidates: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    q = _normalize_ar(user_text)
    if not q or not candidates:
        return None
    best = None
    best_len = 0
    for c in candidates:
        name = _normalize_ar(str(c.get("original_name") or c.get("doc_key") or ""))
        if not name:
            continue
        if q in name and len(q) > best_len:
            best = c
            best_len = len(q)
    return best


# =========================================================
# Download endpoint
# =========================================================
@app.get("/api/files/{public_id}")
def download_file(public_id: str, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.public_id == public_id).first()
    if not doc:
        raise HTTPException(404, "الملف غير موجود.")
    path = getattr(doc, "file_path", None)
    if not path or not os.path.exists(path):
        raise HTTPException(404, "مسار الملف غير موجود على السيرفر.")
    filename = doc.name or os.path.basename(path)
    media_type = _guess_mime(filename)
    return FileResponse(path=path, media_type=media_type, filename=filename)


# =========================================================
# Upload docs + RAG processing  ✅ MULTI FILES
# =========================================================
@app.post("/api/admin/documents/upload")
async def upload_documents(files: List[UploadFile] = File(...), db: Session = Depends(get_db)):
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    results: List[Dict[str, Any]] = []

    for file in (files or []):
        if not file or not (file.filename or "").strip():
            results.append({"status": "error", "message": "اسم الملف فارغ"})
            continue

        existing_doc = db.query(Document).filter(Document.name == file.filename).first()
        if existing_doc:
            results.append(
                {
                    "status": "exists",
                    "filename": file.filename,
                    "message": f"الملف '{file.filename}' موجود مسبقًا ولم يتم الحفظ.",
                }
            )
            try:
                await file.close()
            except Exception:
                pass
            continue

        file_uuid = uuid.uuid4().hex
        safe_name = file.filename.replace("\\", "_").replace("/", "_")
        file_path = os.path.join(upload_dir, f"{file_uuid}_{safe_name}")

        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            file_size_bytes = os.path.getsize(file_path)
            size_str = f"{round(file_size_bytes / (1024 * 1024), 2)} MB"

            new_doc = Document(
                public_id=uuid.uuid4().hex,
                name=file.filename,
                category="جامعة سطام",
                size=size_str,
                upload_date=datetime.utcnow(),
                status="نشط",
                file_path=file_path,
            )
            db.add(new_doc)
            db.commit()

            result = None
            processor_error = None
            try:
                result = await doc_processor.process_file(file_path)
            except Exception as e:
                processor_error = str(e)
                print("⚠️ Document processing failed (keeping original file):", e)

            if result:
                try:
                    converted_path = (result or {}).get("converted_txt_path")
                    converted_name = (result or {}).get("converted_name")
                    if converted_path and converted_name and os.path.exists(converted_path):
                        existing_conv = (
                            db.query(Document).filter(Document.name == converted_name).first()
                        )
                        if not existing_conv:
                            size_bytes2 = os.path.getsize(converted_path)
                            size_str2 = f"{round(size_bytes2 / (1024 * 1024), 2)} MB"
                            conv_doc = Document(
                                public_id=uuid.uuid4().hex,
                                name=converted_name,
                                category="جامعة سطام (HTML)",
                                size=size_str2,
                                upload_date=datetime.utcnow(),
                                status="نشط",
                                file_path=converted_path,
                            )
                            db.add(conv_doc)
                            db.commit()
                except Exception as e:
                    print("⚠️ Failed to save converted HTML doc in DB:", e)

            if processor_error:
                results.append(
                    {
                        "status": "success",
                        "filename": file.filename,
                        "message": f"تم رفع '{file.filename}' ✅ لكن فشل تحويل HTML مؤقتاً: {processor_error}",
                    }
                )
            else:
                results.append(
                    {
                        "status": "success",
                        "filename": file.filename,
                        "message": f"تم رفع ومعالجة '{file.filename}' بنجاح ✅",
                    }
                )

        except Exception as e:
            db.rollback()
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass
            results.append(
                {
                    "status": "error",
                    "filename": file.filename,
                    "message": f"خطأ في رفع الملف: {str(e)}",
                }
            )

        try:
            await file.close()
        except Exception:
            pass

    return {"status": "done", "count": len(results), "results": results}


# =========================================================
# Chat endpoint
# =========================================================
@app.post("/api/chat", response_model=MessageResponse)
async def send_message(message: MessageCreate, db: Session = Depends(get_db)):
    global LAST_TOPIC, PENDING_CHOICES

    # ✅ إذا جاي اختيار زر (doc_key) نخزنّه ونرجع للسؤال الأصلي
    choice_doc_key = (message.choice_doc_key or "").strip()
    forced_doc_key = ""

    if choice_doc_key:
        forced_doc_key = choice_doc_key
        # لو عندنا سؤال أصلي محفوظ، نرجع له عشان نعطي الجواب مباشرة
        pending_q = (PENDING_CHOICES.get("pending_user_text") or "").strip()
        if pending_q:
            user_text = pending_q
        else:
            user_text = (message.content or "").strip()  # fallback
        # نظف حالة الاختيار (بنخلي forced_doc_key يمر لاحقاً)
        LAST_TOPIC = ""
        PENDING_CHOICES["forced_doc_key"] = forced_doc_key
        PENDING_CHOICES["candidates"] = []
    else:
        user_text = (message.content or "").strip()

    if not user_text:
        raise HTTPException(400, "اكتب سؤالك من فضلك.")

    # =========================================================
    # Refine query
    # =========================================================
    refined = llm_service.refine_query(
        user_text,
        context_hint="PSAU University chatbot for academic and administrative questions",
    )

    direct = refined.get("direct_response")
    if direct:
        return MessageResponse(
            id=int(datetime.utcnow().timestamp()),
            content=str(direct),
            sender="assistant",
            timestamp=datetime.utcnow(),
            sources=[],
            attachments=[],
            choices=[],
        )

    request_type = (refined.get("request_type") or "answer").strip().lower()
    file_query = (refined.get("file_query") or "").strip()

    # =========================
    # FILE REQUEST
    # (مهم) إذا عندنا اختيار doc_key من الزر → لا تعاملها كطلب ملف حتى لو المستخدم ضغط اسم ملف
    # =========================
    if request_type == "file" and not forced_doc_key:
        q = file_query.strip() if file_query and len(file_query.strip()) >= 3 else user_text
        doc = _find_best_document(db, q)
        if not doc:
            return MessageResponse(
                id=int(datetime.utcnow().timestamp()),
                content="عذرًا، لم أجد ملفًا مطابقًا لطلبك ضمن الملفات المرفوعة حاليًا. جرّب تكتب كلمات مميزة من اسم الملف.",
                sender="assistant",
                timestamp=datetime.utcnow(),
                sources=[],
                attachments=[],
                choices=[],
            )
        if not getattr(doc, "public_id", None):
            pid = uuid.uuid4().hex
            db.execute(
                text("UPDATE documents SET public_id = :pid WHERE id = :id"),
                {"pid": pid, "id": doc.id},
            )
            db.commit()
            doc.public_id = pid
        attachment = AttachmentOut(
            name=doc.name,
            url=f"/api/files/{doc.public_id}",
            mime=_guess_mime(doc.name or ""),
        )
        return MessageResponse(
            id=int(datetime.utcnow().timestamp()),
            content=f"تفضل، هذا الملف المطلوب: **{doc.name}** ✅",
            sender="assistant",
            timestamp=datetime.utcnow(),
            sources=[],
            attachments=[attachment],
            choices=[],
        )

    # =========================
    # NORMAL ANSWER (RAG)
    # =========================
    refined_question = (refined.get("refined_question") or user_text).strip()
    intent = (refined.get("intent") or "").strip()

    retrieval_query = llm_service.build_retrieval_query_smart(refined, user_text)
    retrieval_query = _fix_ar_spacing(retrieval_query)

    # Follow-up
    short_q = len(user_text.strip()) <= 25
    looks_referential = bool(_PRONOUNY.search(_normalize_ar(user_text)))
    if looks_referential and short_q and LAST_TOPIC and not LAST_TOPIC.endswith("::choose_doc"):
        retrieval_query = _fix_ar_spacing(f"{user_text.strip()} {LAST_TOPIC}".strip())
        print("🔁 Follow-up detected → new retrieval_query:", retrieval_query)

    # Article phrase
    article_phrase = _extract_article_phrase(user_text)
    wants_exec_rules = _is_exec_rules_query(user_text)

    if article_phrase:
        if wants_exec_rules:
            retrieval_query = f"{retrieval_query} {article_phrase} القواعد التنفيذية".strip()
        else:
            retrieval_query = f"{retrieval_query} {article_phrase} نص المادة".strip()
        print("📌 Article phrase detected:", article_phrase, "wants_exec_rules=", wants_exec_rules)

    if len(refined_question) >= 20:
        LAST_TOPIC = refined_question
        print("🧠 LAST_TOPIC updated:", LAST_TOPIC)

    # -------------------------------------------------
    # ✅ Search config (semantic only)
    # -------------------------------------------------
    BROAD_TOP_K = int(os.getenv("RAG_BROAD_TOP_K", "40"))
    FOCUSED_TOP_K = int(os.getenv("RAG_FOCUSED_TOP_K", "10"))
    NEIGHBOR_WINDOW = int(os.getenv("RAG_NEIGHBOR_WINDOW", "2"))
    DIST_THRESHOLD = float(os.getenv("RAG_DISTANCE_THRESHOLD", "0.95"))

    context_docs: List[Dict[str, Any]] = []
    broad_hits: List[Dict[str, Any]] = []
    best_doc_key = ""

    forced_doc_key2 = (
        forced_doc_key
        or str((PENDING_CHOICES or {}).get("forced_doc_key") or "").strip()
    )

    # =========================================================
    # ✅ SEMANTIC ONLY FLOW
    # - broad semantic to guess doc_key
    # - focused semantic with doc_key filter
    # - optional article filtering
    # =========================================================
    try:
        broad_hits = rag_system.search(
            retrieval_query,
            top_k=BROAD_TOP_K,
            include_neighbors=False,
            neighbor_window=0,
        ) or []
    except Exception as e:
        print("RAG broad search error:", e)
        broad_hits = []

    # إذا سؤال مادة: فلترة broad hits بالمادة ثم قرر doc_key
    if article_phrase:
        filtered_article_hits = _filter_hits_for_article(broad_hits, article_phrase)
        candidates = _rank_doc_candidates_from_hits(filtered_article_hits, top_n=6)

        if forced_doc_key2:
            best_doc_key = forced_doc_key2
        else:
            # ✅ إذا أكثر من ملف → رجّع choices أزرار
            if len(candidates) >= 2:
                LAST_TOPIC = f"{article_phrase}::choose_doc"
                PENDING_CHOICES = {
                    "article_phrase": article_phrase,
                    "candidates": candidates,
                    "forced_doc_key": "",
                    "pending_user_text": user_text,  # ✅ نحفظ السؤال الأصلي
                    "ts": datetime.utcnow(),
                }
                choices = [
                    ChoiceOut(label=str(c["original_name"]), doc_key=str(c["doc_key"]))
                    for c in candidates
                ]
                return MessageResponse(
                    id=int(datetime.utcnow().timestamp()),
                    content=(
                        f"لقيت **{article_phrase}** موجودة في أكثر من ملف.\n\n"
                        "أي ملف تقصد؟"
                    ),
                    sender="assistant",
                    timestamp=datetime.utcnow(),
                    sources=[],
                    attachments=[],
                    choices=choices,
                )

            best_doc_key = _pick_best_doc_key_from_hits(filtered_article_hits) or _pick_best_doc_key_from_hits(broad_hits)

    else:
        best_doc_key = forced_doc_key2 or _pick_best_doc_key_from_hits(broad_hits)

    if best_doc_key:
        print("🎯 Auto-selected doc_key:", best_doc_key)

    # Focused semantic search
    try:
        context_docs = rag_system.search(
            retrieval_query,
            top_k=FOCUSED_TOP_K,
            include_neighbors=True,
            neighbor_window=NEIGHBOR_WINDOW,
            where={"doc_key": best_doc_key} if best_doc_key else None,
        ) or []
    except Exception as e:
        print("RAG focused search error:", e)
        context_docs = []

    # Article filter on focused results (semantic-only reinforcement)
    if article_phrase and context_docs:
        filtered2 = _filter_hits_for_article(context_docs, article_phrase)
        if filtered2:
            context_docs = filtered2

    # Fallback: focused without doc_key
    if not context_docs and broad_hits:
        try:
            context_docs = rag_system.search(
                retrieval_query,
                top_k=FOCUSED_TOP_K,
                include_neighbors=True,
                neighbor_window=NEIGHBOR_WINDOW,
            ) or []
        except Exception as e:
            print("RAG fallback search error:", e)
            context_docs = []

        if article_phrase and context_docs:
            filtered3 = _filter_hits_for_article(context_docs, article_phrase)
            if filtered3:
                context_docs = filtered3

    # Threshold gate (لا نطبقه على أسئلة المواد)
    best_score = None
    for d in context_docs:
        sc = d.get("score")
        if isinstance(sc, (int, float)):
            best_score = sc if best_score is None else min(best_score, sc)

    if (not article_phrase) and (best_score is not None) and (best_score > DIST_THRESHOLD):
        print(f"🚫 Best distance {best_score:.4f} > threshold {DIST_THRESHOLD:.4f} → empty context")
        context_docs = []

    # ✅ DEBUG
    print("\n================ RAG DEBUG ================")
    print("user_text:", user_text)
    print("refined_question:", refined_question)
    print("intent:", intent)
    print("retrieval_query:", retrieval_query)
    print("article_phrase:", article_phrase)
    print("wants_exec_rules:", wants_exec_rules)
    print("broad_hits:", len(broad_hits))
    print("best_doc_key:", best_doc_key)
    print("hits:", len(context_docs))
    print("best_score:", best_score)
    for i, d in enumerate(context_docs[:5], 1):
        meta = d.get("metadata") or {}
        content = (d.get("content") or "").strip()
        print(f"\n--- HIT {i} ---")
        print("score:", d.get("score"))
        print("doc_key:", meta.get("doc_key"))
        print("original_name:", meta.get("original_name") or meta.get("filename"))
        print("chunk_index:", meta.get("chunk_index"))
        print("chunk_type:", meta.get("chunk_type"))
        print("section_type:", meta.get("section_type"))
        print("article_no:", meta.get("article_no"))
        print("h2:", meta.get("h2"))
        print("h3:", meta.get("h3"))
        print("is_table:", meta.get("is_table"))
        print("content_len:", len(content))
        print("content_preview:", content[:300].replace("\n", " "))
    print("===========================================\n")

    ai_text = llm_service.generate_response(
        user_query=user_text,
        refined_question=refined_question,
        intent=intent,
        context_docs=context_docs,
        user_role="",
        system_message=None,
    )

    sources = llm_service.extract_sources(
        context_docs or [],
        max_sources=int(os.getenv("MAX_SOURCES", "4")),
    )

    # نظّف forced_doc_key بعد ما نستخدمه
    if PENDING_CHOICES.get("forced_doc_key"):
        PENDING_CHOICES["forced_doc_key"] = ""
    if PENDING_CHOICES.get("pending_user_text"):
        # نخليه فاضي عشان ما يأثر على أسئلة جديدة
        PENDING_CHOICES["pending_user_text"] = ""

    return MessageResponse(
        id=int(datetime.utcnow().timestamp()),
        content=ai_text,
        sender="assistant",
        timestamp=datetime.utcnow(),
        sources=sources,
        attachments=[],
        choices=[],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )