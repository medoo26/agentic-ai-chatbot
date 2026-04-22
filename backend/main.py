# main.py
from __future__ import annotations

import os
import re
import uuid
import shutil
import json
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import or_, text
from sqlalchemy.exc import IntegrityError

from admin_api import router as admin_router
from database import init_db, get_db, AdminUser, Document

from rag_system import RAGSystem
from llm_service import LLMService
from document_processor import DocumentProcessor

load_dotenv()

app = FastAPI(title="PSU Chatbot Backend API", version="1.5.0")

class ChatSessionState:
    def __init__(self):
        self.last_conversation_turn: Dict[str, str] = {"user": "", "assistant": ""}
        self.pending_choices: Dict[str, Any] = {
            "article_phrase": "",
            "candidates": [],
            "forced_doc_key": "",
            "pending_user_text": "",
            "ts": None,
        }


_SESSION_STATES: Dict[str, ChatSessionState] = {}
_SESSION_LOCK = asyncio.Lock()

_PRONOUNY = re.compile(
    r"(اهدافها|مهامها|شروطها|مواعيدها|رسومها|آليتها|طريقتها|كيفها|وش هي|وشو|هذي|هذا|تلك|ذي)\b"
)


_NO_INFO_RE = re.compile(
    r"عذراً[،,]?\s*.*?غير\s+متوفرة\s+في\s+الملفات\s+المرفوعة\s+حالياً",
    re.IGNORECASE,
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
            try:
                db.commit()
                print(f"✅ Default admin created: {admin_username}")
            except IntegrityError:
                # Uvicorn reload can spawn multiple startup processes concurrently.
                # If another process created the same admin first, ignore safely.
                db.rollback()
    finally:
        db.close()


init_db()
create_default_admin()


# =========================================================
# Schemas
# =========================================================
class MessageCreate(BaseModel):
    content: str
    choice_doc_key: Optional[str] = None
    session_id: Optional[str] = None


class AttachmentOut(BaseModel):
    name: str
    url: str
    mime: Optional[str] = None


class ChoiceOut(BaseModel):
    label: str
    doc_key: str


class SourceOut(BaseModel):
    name: str
    page: Optional[int] = None
    public_id: Optional[str] = None
    url: Optional[str] = None
    mime: Optional[str] = None


class MessageResponse(BaseModel):
    id: int
    content: str
    sender: str
    timestamp: datetime
    sources: List[SourceOut] = []
    attachments: List[AttachmentOut] = []
    choices: List[ChoiceOut] = []
    debug: Optional[str] = None


def _coerce_session_id(raw: str) -> str:
    v = (raw or "").strip()
    if not v:
        return uuid.uuid4().hex
    return re.sub(r"[^a-zA-Z0-9_\-]", "", v)[:64] or uuid.uuid4().hex


async def _get_session_state(request: Request, message: MessageCreate) -> ChatSessionState:
    header_sid = request.headers.get("x-chat-session-id", "")
    body_sid = (message.session_id or "").strip()
    session_id = _coerce_session_id(header_sid or body_sid)
    request.state.chat_session_id = session_id
    async with _SESSION_LOCK:
        state = _SESSION_STATES.get(session_id)
        if state is None:
            state = ChatSessionState()
            _SESSION_STATES[session_id] = state
        return state


def _sse_event(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _is_no_info_answer(text: str) -> bool:
    return bool(_NO_INFO_RE.search((text or "").strip()))


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
# Arabic text helpers
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

    # Prefer original uploaded documents; exclude converted processed_html artifacts.
    docs_q = (
        db.query(Document)
        .filter(~Document.name.ilike("%__html.txt"))
        .filter(~Document.category.ilike("%(HTML)%"))
        .filter(
            or_(
                Document.file_path.ilike("uploads/%"),
                Document.file_path.ilike("uploads\\%"),
                Document.file_path.ilike("%/uploads/%"),
                Document.file_path.ilike("%\\uploads\\%"),
            )
        )
    )

    q_norm = _normalize_ar(q)
    if any(
        ext in q_norm
        for ext in [".pdf", ".docx", ".doc", ".txt", ".xlsx", ".pptx", ".ppt"]
    ):
        doc = (
            docs_q
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
            docs_q
            .filter(Document.name.ilike(f"%{short}%"))
            .order_by(Document.upload_date.desc())
            .first()
        )

    conditions = [Document.name.ilike(f"%{k}%") for k in keys]
    return (
        docs_q
        .filter(or_(*conditions))
        .order_by(Document.upload_date.desc())
        .first()
    )


# =========================================================
# Auto-select best doc_key from hits
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


def _extract_sources_with_pages(
    hits: List[Dict[str, Any]],
    max_sources: int = 6,
    max_pages_per_file: int = 2,
    db: Optional[Session] = None,
) -> List[SourceOut]:
    def _source_lookup_key(name: str) -> str:
        n = (name or "").strip().lower()
        n = re.sub(r"\.(pdf|docx|doc|txt|xlsx|xls|pptx|ppt)$", "", n, flags=re.IGNORECASE)
        n = re.sub(r"\s*\(\d+\)\s*$", "", n)
        n = _normalize_ar(n)
        n = re.sub(r"\s+", " ", n).strip()
        return n

    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for h in hits or []:
        meta = h.get("metadata") or {}
        name = str(
            meta.get("original_name")
            or meta.get("filename")
            or meta.get("doc_key")
            or ""
        ).strip()
        if not name:
            continue

        page_number = h.get("page_number")
        if page_number is None:
            page_number = meta.get("page_number")

        try:
            page_number = int(page_number) if page_number is not None else None
        except Exception:
            page_number = None

        score = h.get("score")
        try:
            score = float(score) if score is not None else 1e9
        except Exception:
            score = 1e9

        grouped.setdefault(name, []).append(
            {
                "name": name,
                "page": page_number,
                "score": score,
            }
        )

    doc_map: Dict[str, Document] = {}
    doc_key_map: Dict[str, Document] = {}
    if db is not None and grouped:
        docs = db.query(Document).all()
        for d in docs:
            key = str(getattr(d, "name", "") or "").strip()
            if key and key not in doc_map:
                doc_map[key] = d
            nk = _source_lookup_key(key)
            if nk and nk not in doc_key_map:
                doc_key_map[nk] = d

    out: List[SourceOut] = []

    for name, items in grouped.items():
        items.sort(key=lambda x: x["score"])

        seen_pages = set()
        picked = 0

        for item in items:
            page = item["page"]
            if page in seen_pages:
                continue
            seen_pages.add(page)

            doc = doc_map.get(name) or doc_key_map.get(_source_lookup_key(name))
            public_id = ""
            url = None
            mime = None
            if doc:
                public_id = str(getattr(doc, "public_id", "") or "").strip()
                if not public_id:
                    public_id = uuid.uuid4().hex
                    db.execute(
                        text("UPDATE documents SET public_id = :pid WHERE id = :id"),
                        {"pid": public_id, "id": doc.id},
                    )
                    db.commit()
                    doc.public_id = public_id
                url = f"/api/files/{public_id}/preview"
                mime = _guess_mime(str(getattr(doc, "name", "") or ""))

            out.append(
                SourceOut(
                    name=name,
                    page=page,
                    public_id=public_id or None,
                    url=url,
                    mime=mime,
                )
            )
            picked += 1

            if picked >= max_pages_per_file:
                break

    out = out[:max_sources]
    return out


def _extract_sources_from_llm_titles(
    llm_titles: List[str],
    hits: List[Dict[str, Any]],
    max_sources: int = 6,
    max_pages_per_file: int = 2,
    db: Optional[Session] = None,
) -> List[SourceOut]:
    def _source_lookup_key(name: str) -> str:
        n = (name or "").strip().lower()
        n = re.sub(r"\.(pdf|docx|doc|txt|xlsx|xls|pptx|ppt)$", "", n, flags=re.IGNORECASE)
        n = re.sub(r"\s*\(\d+\)\s*$", "", n)
        n = _normalize_ar(n)
        n = re.sub(r"\s+", " ", n).strip()
        return n

    titles_norm = [_normalize_ar(str(t or "").strip()) for t in (llm_titles or []) if str(t or "").strip()]
    titles_norm = [t for t in titles_norm if t]
    if not titles_norm:
        return _extract_sources_with_pages(
            hits or [],
            max_sources=max_sources,
            max_pages_per_file=max_pages_per_file,
            db=db,
        )

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for h in hits or []:
        meta = h.get("metadata") or {}
        name = str(
            meta.get("original_name")
            or meta.get("filename")
            or meta.get("doc_key")
            or ""
        ).strip()
        if not name:
            continue

        page_number = h.get("page_number")
        if page_number is None:
            page_number = meta.get("page_number")
        try:
            page_number = int(page_number) if page_number is not None else None
        except Exception:
            page_number = None

        score = h.get("score")
        try:
            score = float(score) if score is not None else 1e9
        except Exception:
            score = 1e9

        grouped.setdefault(name, []).append({"page": page_number, "score": score})

    selected_names: List[str] = []
    for name in grouped.keys():
        nn = _normalize_ar(name)
        if any((t in nn) or (nn in t) for t in titles_norm):
            selected_names.append(name)

    if not selected_names:
        return _extract_sources_with_pages(
            hits or [],
            max_sources=max_sources,
            max_pages_per_file=max_pages_per_file,
            db=db,
        )

    doc_map: Dict[str, Document] = {}
    doc_key_map: Dict[str, Document] = {}
    if db is not None and grouped:
        docs = db.query(Document).all()
        for d in docs:
            key = str(getattr(d, "name", "") or "").strip()
            if key and key not in doc_map:
                doc_map[key] = d
            nk = _source_lookup_key(key)
            if nk and nk not in doc_key_map:
                doc_key_map[nk] = d

    out: List[SourceOut] = []
    for name in selected_names:
        items = grouped.get(name, [])
        items.sort(key=lambda x: x["score"])
        seen_pages = set()
        picked = 0
        for item in items:
            page = item["page"]
            if page in seen_pages:
                continue
            seen_pages.add(page)
            doc = doc_map.get(name) or doc_key_map.get(_source_lookup_key(name))
            public_id = ""
            url = None
            mime = None
            if doc:
                public_id = str(getattr(doc, "public_id", "") or "").strip()
                if not public_id:
                    public_id = uuid.uuid4().hex
                    db.execute(
                        text("UPDATE documents SET public_id = :pid WHERE id = :id"),
                        {"pid": public_id, "id": doc.id},
                    )
                    db.commit()
                    doc.public_id = public_id
                url = f"/api/files/{public_id}/preview"
                mime = _guess_mime(str(getattr(doc, "name", "") or ""))
            out.append(
                SourceOut(
                    name=name,
                    page=page,
                    public_id=public_id or None,
                    url=url,
                    mime=mime,
                )
            )
            picked += 1
            if picked >= max_pages_per_file:
                break
        if len(out) >= max_sources:
            break

    return out[:max_sources]


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


@app.get("/api/files/{public_id}/preview")
def preview_file(public_id: str, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.public_id == public_id).first()
    if not doc:
        raise HTTPException(404, "الملف غير موجود.")
    path = getattr(doc, "file_path", None)
    if not path or not os.path.exists(path):
        raise HTTPException(404, "مسار الملف غير موجود على السيرفر.")

    filename = doc.name or os.path.basename(path)
    media_type = _guess_mime(filename)
    if media_type != "application/pdf":
        raise HTTPException(400, "المعاينة متاحة لملفات PDF فقط.")

    return FileResponse(
        path=path,
        media_type="application/pdf",
        headers={
            # Keep inline preview header ASCII-only to avoid latin-1 encoding
            # failures with Arabic filenames in Starlette response headers.
            "Content-Disposition": "inline",
            "Accept-Ranges": "bytes",
        },
    )


# =========================================================
# Upload docs + RAG processing  MULTI FILES
# =========================================================
@app.post("/api/admin/documents/upload")
async def upload_documents(files: List[UploadFile] = File(...), db: Session = Depends(get_db)):
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    results: List[Dict[str, Any]] = []
    print(f"[TRACE][UPLOAD] start | files={len(files or [])}")

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

        safe_name = file.filename.replace("\\", "_").replace("/", "_")
        file_path = os.path.join(upload_dir, safe_name)

        try:
            print(f"[TRACE][UPLOAD] save_start | file={file.filename}")
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
                print(f"[TRACE][UPLOAD] process_start | file={file.filename} path={file_path}")
                result = await doc_processor.process_file(file_path)
                print(f"[TRACE][UPLOAD] process_done | file={file.filename}")
            except Exception as e:
                processor_error = str(e)
                print("⚠️ Document processing failed (keeping original file):", e)
                print(f"[TRACE][UPLOAD] process_error | file={file.filename} error={processor_error}")

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
            print(f"[TRACE][UPLOAD] file_error | file={file.filename} error={e}")

        try:
            await file.close()
        except Exception:
            pass

    print(f"[TRACE][UPLOAD] done | files={len(files or [])} results={len(results)}")
    return {"status": "done", "count": len(results), "results": results}


# =========================================================
# Chat endpoint
# =========================================================
@app.post("/api/chat", response_model=MessageResponse)
async def send_message(
    message: MessageCreate, request: Request, db: Session = Depends(get_db)
):
    session_state = await _get_session_state(request, message)
    pending_choices = session_state.pending_choices
    previous_turn = session_state.last_conversation_turn

    choice_doc_key = (message.choice_doc_key or "").strip()
    forced_doc_key = ""

    if choice_doc_key:
        forced_doc_key = choice_doc_key
        pending_q = (pending_choices.get("pending_user_text") or "").strip()
        if pending_q:
            user_text = pending_q
        else:
            user_text = (message.content or "").strip()
        pending_choices["forced_doc_key"] = forced_doc_key
        pending_choices["candidates"] = []
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
        previous_turn=previous_turn,
    )
    print(
        "[TRACE][CHAT] refine_query",
        {
            "user_text": user_text,
            "request_type": refined.get("request_type"),
            "file_query": refined.get("file_query"),
            "refined_question": refined.get("refined_question"),
            "is_followup": refined.get("is_followup"),
            "direct_response": refined.get("direct_response"),
        },
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
            debug=None,
        )

    request_type = (refined.get("request_type") or "answer").strip().lower()
    file_query = (refined.get("file_query") or "").strip()
    is_followup = bool(refined.get("is_followup"))

    # =========================
    # FILE REQUEST
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
                debug=None,
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
            debug=None,
        )

    # =========================
    # NORMAL ANSWER (RAG)
    # =========================
    refined_question = (refined.get("refined_question") or user_text).strip()
    retrieval_query = refined_question or user_text


    # -------------------------------------------------
    # Search config (semantic only)
    # -------------------------------------------------
    BROAD_TOP_K = int(os.getenv("RAG_BROAD_TOP_K", "40"))
    FOCUSED_TOP_K = int(os.getenv("RAG_FOCUSED_TOP_K", "20"))
    NEIGHBOR_WINDOW = int(os.getenv("RAG_NEIGHBOR_WINDOW", "2"))
    ENABLE_NEIGHBORS_FOR_CONTEXT = (os.getenv("RAG_ENABLE_NEIGHBORS_FOR_CONTEXT", "1").strip() == "1")

    context_docs: List[Dict[str, Any]] = []
    source_hits: List[Dict[str, Any]] = []
    broad_hits: List[Dict[str, Any]] = []

    # =========================================================
    # SEMANTIC ONLY FLOW
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

    # -------------------------------------------------
    # مصدر العرض: بدون neighbors
    # -------------------------------------------------
    try:
        source_hits = rag_system.search(
            retrieval_query,
            top_k=FOCUSED_TOP_K,
            include_neighbors=False,
            neighbor_window=0,
        ) or []
    except Exception as e:
        print("RAG source search error:", e)
        source_hits = []

    # -------------------------------------------------
    # سياق الجواب: مع neighbors
    # -------------------------------------------------
    try:
        context_docs = rag_system.search(
            retrieval_query,
            top_k=FOCUSED_TOP_K,
            include_neighbors=ENABLE_NEIGHBORS_FOR_CONTEXT,
            neighbor_window=NEIGHBOR_WINDOW if ENABLE_NEIGHBORS_FOR_CONTEXT else 0,
        ) or []
    except Exception as e:
        print("RAG focused search error:", e)
        context_docs = []

    # Fallback: reuse broad hits instead of repeating identical searches
    if not context_docs and broad_hits:
        if not source_hits:
            source_hits = broad_hits[:FOCUSED_TOP_K]
        context_docs = broad_hits[:FOCUSED_TOP_K]

    debug_text = llm_service._build_context(context_docs or [])

    ai_text = llm_service.generate_response(
        user_query=user_text,
        refined_question=refined_question,
        context_docs=context_docs,
        user_role="",
        system_message=None,
        previous_turn=previous_turn if is_followup else None,
    )

    session_state.last_conversation_turn = {
        "user": user_text,
        "assistant": ai_text,
    }

    sources = _extract_sources_from_llm_titles(
        getattr(llm_service, "last_answer_source_titles", []) or [],
        source_hits or context_docs or [],
        max_sources=int(os.getenv("MAX_SOURCES", "6")),
        max_pages_per_file=int(os.getenv("MAX_PAGES_PER_FILE", "2")),
        db=db,
    )
    if _is_no_info_answer(ai_text):
        sources = []

    if pending_choices.get("forced_doc_key"):
        pending_choices["forced_doc_key"] = ""
    if pending_choices.get("pending_user_text"):
        pending_choices["pending_user_text"] = ""


    return MessageResponse(
        id=int(datetime.utcnow().timestamp()),
        content=ai_text,
        sender="assistant",
        timestamp=datetime.utcnow(),
        sources=sources,
        attachments=[],
        choices=[],
        debug=debug_text,
    )


@app.post("/api/chat/stream")
async def stream_message(
    message: MessageCreate, request: Request, db: Session = Depends(get_db)
):
    session_state = await _get_session_state(request, message)
    pending_choices = session_state.pending_choices
    previous_turn = session_state.last_conversation_turn

    choice_doc_key = (message.choice_doc_key or "").strip()
    forced_doc_key = ""
    if choice_doc_key:
        forced_doc_key = choice_doc_key
        pending_q = (pending_choices.get("pending_user_text") or "").strip()
        user_text = pending_q if pending_q else (message.content or "").strip()
        pending_choices["forced_doc_key"] = forced_doc_key
        pending_choices["candidates"] = []
    else:
        user_text = (message.content or "").strip()

    if not user_text:
        raise HTTPException(400, "اكتب سؤالك من فضلك.")

    refined = llm_service.refine_query(
        user_text,
        context_hint="PSAU University chatbot for academic and administrative questions",
        previous_turn=previous_turn,
    )
    direct = refined.get("direct_response")
    request_type = (refined.get("request_type") or "answer").strip().lower()
    file_query = (refined.get("file_query") or "").strip()
    is_followup = bool(refined.get("is_followup"))

    async def event_stream():
        try:
            if direct:
                content = str(direct)
                for chunk in llm_service._iter_text_deltas(content):
                    if await request.is_disconnected():
                        return
                    yield _sse_event("token", {"delta": chunk})
                yield _sse_event(
                    "meta",
                    {
                        "id": int(datetime.utcnow().timestamp()),
                        "timestamp": datetime.utcnow().isoformat(),
                        "sources": [],
                        "attachments": [],
                        "choices": [],
                        "debug": None,
                    },
                )
                yield _sse_event("done", {"ok": True})
                return

            if request_type == "file" and not forced_doc_key:
                q = file_query.strip() if file_query and len(file_query.strip()) >= 3 else user_text
                doc = _find_best_document(db, q)
                if not doc:
                    not_found = "عذرًا، لم أجد ملفًا مطابقًا لطلبك ضمن الملفات المرفوعة حاليًا. جرّب تكتب كلمات مميزة من اسم الملف."
                    for chunk in llm_service._iter_text_deltas(not_found):
                        if await request.is_disconnected():
                            return
                        yield _sse_event("token", {"delta": chunk})
                    yield _sse_event(
                        "meta",
                        {
                            "id": int(datetime.utcnow().timestamp()),
                            "timestamp": datetime.utcnow().isoformat(),
                            "sources": [],
                            "attachments": [],
                            "choices": [],
                            "debug": None,
                        },
                    )
                    yield _sse_event("done", {"ok": True})
                    return

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
                file_text = f"تفضل، هذا الملف المطلوب: **{doc.name}** ✅"
                for chunk in llm_service._iter_text_deltas(file_text):
                    if await request.is_disconnected():
                        return
                    yield _sse_event("token", {"delta": chunk})
                yield _sse_event(
                    "meta",
                    {
                        "id": int(datetime.utcnow().timestamp()),
                        "timestamp": datetime.utcnow().isoformat(),
                        "sources": [],
                        "attachments": [attachment.model_dump()],
                        "choices": [],
                        "debug": None,
                    },
                )
                yield _sse_event("done", {"ok": True})
                return

            refined_question = (refined.get("refined_question") or user_text).strip()
            retrieval_query = refined_question or user_text


            BROAD_TOP_K = int(os.getenv("RAG_BROAD_TOP_K", "40"))
            FOCUSED_TOP_K = int(os.getenv("RAG_FOCUSED_TOP_K", "20"))
            NEIGHBOR_WINDOW = int(os.getenv("RAG_NEIGHBOR_WINDOW", "2"))
            ENABLE_NEIGHBORS_FOR_CONTEXT = (os.getenv("RAG_ENABLE_NEIGHBORS_FOR_CONTEXT", "1").strip() == "1")

            context_docs: List[Dict[str, Any]] = []
            source_hits: List[Dict[str, Any]] = []
            broad_hits: List[Dict[str, Any]] = []

            try:
                broad_hits = rag_system.search(
                    retrieval_query,
                    top_k=BROAD_TOP_K,
                    include_neighbors=False,
                    neighbor_window=0,
                ) or []
            except Exception:
                broad_hits = []

            try:
                source_hits = rag_system.search(
                    retrieval_query,
                    top_k=FOCUSED_TOP_K,
                    include_neighbors=False,
                    neighbor_window=0,
                ) or []
            except Exception:
                source_hits = []

            try:
                context_docs = rag_system.search(
                    retrieval_query,
                    top_k=FOCUSED_TOP_K,
                    include_neighbors=ENABLE_NEIGHBORS_FOR_CONTEXT,
                    neighbor_window=NEIGHBOR_WINDOW if ENABLE_NEIGHBORS_FOR_CONTEXT else 0,
                ) or []
            except Exception:
                context_docs = []

            if not context_docs and broad_hits:
                if not source_hits:
                    source_hits = broad_hits[:FOCUSED_TOP_K]
                context_docs = broad_hits[:FOCUSED_TOP_K]

            parts: List[str] = []
            for delta in llm_service.generate_response_stream(
                user_query=user_text,
                refined_question=refined_question,
                context_docs=context_docs,
                user_role="",
                system_message=None,
                previous_turn=previous_turn if is_followup else None,
            ):
                if await request.is_disconnected():
                    return
                parts.append(delta)
                yield _sse_event("token", {"delta": delta})

            ai_text = "".join(parts).strip()
            session_state.last_conversation_turn = {"user": user_text, "assistant": ai_text}
            if pending_choices.get("forced_doc_key"):
                pending_choices["forced_doc_key"] = ""
            if pending_choices.get("pending_user_text"):
                pending_choices["pending_user_text"] = ""

            sources = _extract_sources_from_llm_titles(
                getattr(llm_service, "last_answer_source_titles", []) or [],
                source_hits or context_docs or [],
                max_sources=int(os.getenv("MAX_SOURCES", "6")),
                max_pages_per_file=int(os.getenv("MAX_PAGES_PER_FILE", "2")),
                db=db,
            )
            if _is_no_info_answer(ai_text):
                sources = []
            debug_text = llm_service._build_context(context_docs or [])
            yield _sse_event(
                "meta",
                {
                    "id": int(datetime.utcnow().timestamp()),
                    "timestamp": datetime.utcnow().isoformat(),
                    "sources": [s.model_dump() for s in sources],
                    "attachments": [],
                    "choices": [],
                    "debug": debug_text,
                },
            )
            yield _sse_event("done", {"ok": True})
        except Exception as exc:
            yield _sse_event("error", {"message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )