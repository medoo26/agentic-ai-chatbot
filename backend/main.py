# main.py
from fastapi import FastAPI, Depends, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional, Any, Dict
from pydantic import BaseModel
from datetime import datetime
import os
import uuid
import shutil
from dotenv import load_dotenv
from passlib.context import CryptContext

# استيراد المكونات المحلية للمشروع
from admin_api import router as admin_router
from database import (
    init_db,
    get_db,
    SystemSettings,
    AdminUser,
    Document,
)

from rag_system import RAGSystem
from llm_service import LLMService
from document_processor import DocumentProcessor

load_dotenv()

app = FastAPI(title="PSU Chatbot Backend API", version="1.4.0")

# تضمين راوتر الإدارة
app.include_router(admin_router)

# إعدادات CORS (بدون كوكيز/credentials)
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in cors_origins if o.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# تهيئة الخدمات
rag_system = RAGSystem()
llm_service = LLMService()
doc_processor = DocumentProcessor(rag_system)

# =========================================================
# Ensure SystemSettings + Default Admin
# =========================================================
def ensure_system_settings():
    db = next(get_db())
    try:
        s = db.query(SystemSettings).first()
        if not s:
            db.add(SystemSettings())
            db.commit()
            print("✅ SystemSettings created.")
    finally:
        db.close()


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


# init
init_db()
ensure_system_settings()
create_default_admin()

# =========================================================
# Schemas
# =========================================================
class MessageCreate(BaseModel):
    content: str


class AttachmentOut(BaseModel):
    id: int
    name: str
    url: str
    mime: Optional[str] = None


class MessageResponse(BaseModel):
    id: int
    content: str
    sender: str
    timestamp: datetime
    sources: List[str] = []
    attachments: List[AttachmentOut] = []  # ✅ جديد


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


def _find_best_document(db: Session, query: str) -> Optional[Document]:
    """
    بحث بسيط وآمن عن أقرب ملف في جدول documents بالاسم.
    """
    q = (query or "").strip()
    if not q:
        return None

    # بحث contains
    candidates = (
        db.query(Document)
        .filter(Document.name.contains(q))
        .order_by(Document.upload_date.desc())
        .limit(10)
        .all()
    )

    if candidates:
        return candidates[0]

    # fallback: جرّب كلمات منفصلة (أكثر مرونة)
    words = [w.strip() for w in q.split() if w.strip()]
    for w in words[:3]:
        candidates = (
            db.query(Document)
            .filter(Document.name.contains(w))
            .order_by(Document.upload_date.desc())
            .limit(10)
            .all()
        )
        if candidates:
            return candidates[0]

    return None


# =========================================================
# Health / status
# =========================================================
@app.get("/api/health")
async def health_check():
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    return {
        "status": "ok",
        "openai_available": bool(openai_key),
        "gemini_available": bool(gemini_key),
        "vector_store_ready": True,
        "refiner_provider": os.getenv("REFINER_PROVIDER", "gemini"),
        "refine_model": os.getenv("REFINE_MODEL", "gemini-1.5-flash"),
        "chat_model": os.getenv("CHAT_MODEL", "gpt-4o-mini"),
    }


@app.get("/api/admin/llm-status")
async def llm_status():
    return {
        "llm_available": llm_service.is_available(),
        "provider": "OpenAI",
        "model_name": os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        "api_key_set": bool((os.getenv("OPENAI_API_KEY") or "").strip()),
        "refiner_provider": os.getenv("REFINER_PROVIDER", "gemini"),
        "refine_model": os.getenv("REFINE_MODEL", "gemini-1.5-flash"),
        "gemini_key_set": bool((os.getenv("GEMINI_API_KEY") or "").strip()),
    }


# =========================================================
# Download endpoint (new)
# =========================================================
@app.get("/api/files/{doc_id}")
def download_file(doc_id: int, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(404, "الملف غير موجود.")

    path = getattr(doc, "file_path", None)
    if not path or not os.path.exists(path):
        raise HTTPException(404, "مسار الملف غير موجود على السيرفر.")

    filename = doc.name or os.path.basename(path)
    media_type = _guess_mime(filename)

    # FileResponse يرسل الملف للتنزيل مباشرة
    return FileResponse(
        path=path,
        media_type=media_type,
        filename=filename,
    )


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

    file_id = str(uuid.uuid4())
    safe_name = file.filename.replace("\\", "_").replace("/", "_")
    file_path = os.path.join(upload_dir, f"{file_id}_{safe_name}")

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size_bytes = os.path.getsize(file_path)
        size_str = f"{round(file_size_bytes / (1024 * 1024), 2)} MB"

        new_doc = Document(
            name=file.filename,
            category="جامعة سطام",
            size=size_str,
            upload_date=datetime.utcnow(),
            status="نشط",
            file_path=file_path
        )
        db.add(new_doc)
        db.commit()

        await doc_processor.process_file(file_path)

        return {"status": "success", "message": f"تم رفع ومعالجة '{file.filename}' بنجاح."}

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"خطأ في رفع الملف: {str(e)}")


# =========================================================
# ✅ Chat endpoint (STATELESS) - لا يحفظ شيء
# =========================================================
@app.post("/api/chat", response_model=MessageResponse)
async def send_message(
    message: MessageCreate,
    db: Session = Depends(get_db),
):
    user_text = (message.content or "").strip()
    if not user_text:
        raise HTTPException(400, "اكتب سؤالك من فضلك.")

    settings = db.query(SystemSettings).first()
    if settings and not settings.chatbot_available:
        return MessageResponse(
            id=-1,
            content="المساعد متوقف حالياً من قبل الإدارة.",
            sender="assistant",
            timestamp=datetime.utcnow(),
            sources=[],
            attachments=[],
        )

    # Refine
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

    # ✅ NEW: إذا المستخدم يطلب ملف -> رجّع رابط تنزيل بدل إجابة طويلة
    request_type = (refined.get("request_type") or "answer").strip().lower()
    file_query = (refined.get("file_query") or "").strip()

    if request_type == "file":
        # إذا ما عندنا file_query واضح، استخدم السؤال نفسه
        q = file_query or user_text

        doc = _find_best_document(db, q)

        if not doc:
            # رجّع رد لطيف بدون تخمين
            return MessageResponse(
                id=int(datetime.utcnow().timestamp()),
                content="عذرًا، لم أجد ملفًا مطابقًا لطلبك ضمن الملفات المرفوعة حاليًا. جرّب تكتب اسم النموذج/الدليل بشكل أدق.",
                sender="assistant",
                timestamp=datetime.utcnow(),
                sources=[],
                attachments=[],
            )

        attachment = AttachmentOut(
            id=doc.id,
            name=doc.name,
            url=f"/api/files/{doc.id}",
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

    # ====== عادي: إجابة نصية ======
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
        system_message=settings.system_message if settings else None,
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
