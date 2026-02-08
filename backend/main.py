# main.py
from __future__ import annotations

import os
import re
import uuid
import shutil
from datetime import datetime
from typing import List, Optional

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

from rag_system import RAGSystem
from llm_service import LLMService
from document_processor import DocumentProcessor

load_dotenv()

app = FastAPI(title="PSU Chatbot Backend API", version="1.4.3")

# Admin router (documents list/delete/auth ...)
app.include_router(admin_router)

# CORS
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
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
doc_processor = DocumentProcessor(rag_system)

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

# init (✅ migration صار داخل database.init_db() خلاص)
init_db()
create_default_admin()

# =========================================================
# Schemas
# =========================================================
class MessageCreate(BaseModel):
    content: str

class AttachmentOut(BaseModel):
    name: str
    url: str
    mime: Optional[str] = None

class MessageResponse(BaseModel):
    id: int
    content: str
    sender: str
    timestamp: datetime
    sources: List[str] = []
    attachments: List[AttachmentOut] = []

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
    "من", "في", "على", "الى", "إلى", "عن", "مع", "و", "او", "أو", "هذا", "هذه",
    "عطني", "ابغى", "أبغى", "ارسل", "أرسل", "نموذج", "ملف", "تحميل", "رابط",
    "لو", "ليه", "وش", "ايش", "وشو", "كيف", "ممكن", "فضلا", "فضلاً"
}


def _normalize_ar(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    s = s.replace("ة", "ه").replace("ى", "ي")
    return s


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

    if any(ext in q_norm for ext in [".pdf", ".docx", ".doc", ".txt", ".xlsx", ".pptx", ".ppt"]):
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
# Download endpoint (✅ FIXED: use public_id)
# =========================================================
@app.get("/api/files/{public_id}")
def download_file(public_id: str, db: Session = Depends(get_db)):
    # مهم: هنا نعتمد على ORM لأن العمود صار موجود بعد migration
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
# Upload docs + RAG processing (dedupe)
# =========================================================
@app.post("/api/admin/documents/upload")
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    existing_doc = db.query(Document).filter(Document.name == file.filename).first()
    if existing_doc:
        return {"status": "exists", "message": f"الملف '{file.filename}' موجود مسبقًا ولم يتم الحفظ."}

    file_uuid = uuid.uuid4().hex
    safe_name = file.filename.replace("\\", "_").replace("/", "_")
    file_path = os.path.join(upload_dir, f"{file_uuid}_{safe_name}")

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size_bytes = os.path.getsize(file_path)
        size_str = f"{round(file_size_bytes / (1024 * 1024), 2)} MB"

        new_doc = Document(
            public_id=uuid.uuid4().hex,  # ✅ ثابت للتحميل
            name=file.filename,
            category="جامعة سطام",
            size=size_str,
            upload_date=datetime.utcnow(),
            status="نشط",
            file_path=file_path,
        )
        db.add(new_doc)
        db.commit()

        await doc_processor.process_file(file_path)

        return {"status": "success", "message": f"تم رفع ومعالجة '{file.filename}' بنجاح."}

    except Exception as e:
        db.rollback()
        # حاول تحذف الملف لو انكتب
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"خطأ في رفع الملف: {str(e)}")


# =========================================================
# Chat endpoint (STATELESS)
# =========================================================
@app.post("/api/chat", response_model=MessageResponse)
async def send_message(message: MessageCreate, db: Session = Depends(get_db)):
    user_text = (message.content or "").strip()
    if not user_text:
        raise HTTPException(400, "اكتب سؤالك من فضلك.")

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
        )

    request_type = (refined.get("request_type") or "answer").strip().lower()
    file_query = (refined.get("file_query") or "").strip()

    # =========================
    # FILE REQUEST
    # =========================
    if request_type == "file":
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
            )

        # ✅ ضمان وجود public_id حتى لو doc قديم (باستخدام SQL خام)
        if not getattr(doc, "public_id", None):
            pid = uuid.uuid4().hex
            db.execute(
                text("UPDATE documents SET public_id = :pid WHERE id = :id"),
                {"pid": pid, "id": doc.id},
            )
            db.commit()
            doc.public_id = pid  # حدّث الكائن الحالي

        attachment = AttachmentOut(
            name=doc.name,
            url=f"/api/files/{doc.public_id}",  # ✅ ثابت
            mime=_guess_mime(doc.name or ""),
        )

        return MessageResponse(
            id=int(datetime.utcnow().timestamp()),
            content=f"تفضل، هذا الملف المطلوب: **{doc.name}** ✅",
            sender="assistant",
            timestamp=datetime.utcnow(),
            sources=[],
            attachments=[attachment],
        )

    # =========================
    # NORMAL ANSWER (RAG)
    # =========================
    refined_question = (refined.get("refined_question") or user_text).strip()
    intent = (refined.get("intent") or "").strip()
    retrieval_query = llm_service.build_retrieval_query(refined) or refined_question or user_text

    try:
        context_docs = rag_system.search(retrieval_query, top_k=3) or []
    except Exception as e:
        print("RAG search error:", e)
        context_docs = []

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

    return MessageResponse(
        id=int(datetime.utcnow().timestamp()),
        content=ai_text,
        sender="assistant",
        timestamp=datetime.utcnow(),
        sources=sources,
        attachments=[],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )
