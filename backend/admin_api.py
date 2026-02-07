from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import os

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

from passlib.context import CryptContext

from database import (
    get_db,
    Document,
    DocumentChunk,
    KnowledgeEntry,
    SystemSettings,
    AdminUser,
)

# =========================================================
# Auth
# =========================================================
security = HTTPBasic()
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def verify_admin(
    credentials: HTTPBasicCredentials = Depends(security),
    db: Session = Depends(get_db),
):
    username = (credentials.username or "").strip()
    password = (credentials.password or "").strip()

    admin = db.query(AdminUser).filter(AdminUser.username == username).first()
    if not admin or not pwd_context.verify(password, admin.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return admin


def safe_count(db: Session, model):
    try:
        return db.query(func.count(model.id)).scalar() or 0
    except Exception:
        return 0


# =========================================================
# Router
# =========================================================
router = APIRouter(prefix="/api/admin", tags=["admin"])


# =========================================================
# Helpers
# =========================================================
def parse_size_to_mb(size: Optional[str]) -> Optional[float]:
    if not size:
        return None
    s = str(size).strip().lower()
    try:
        parts = s.replace(",", "").split()
        num = float(parts[0])
        unit = parts[1] if len(parts) > 1 else ""
        if unit in ("mb", "mib"):
            return round(num, 2)
        if unit in ("kb", "kib"):
            return round(num / 1024.0, 2)
        if unit in ("gb", "gib"):
            return round(num * 1024.0, 2)
    except Exception:
        return None
    return None


def doc_status_to_bool(status_val: Optional[str]) -> bool:
    return (status_val or "نشط").strip() == "نشط"


def bool_to_doc_status(is_active: bool) -> str:
    return "نشط" if is_active else "غير نشط"


def safe_remove_file(path: Optional[str]) -> bool:
    """يحذف الملف من الهارد لو موجود."""
    if not path:
        return False
    try:
        if os.path.exists(path) and os.path.isfile(path):
            os.remove(path)
            return True
    except Exception:
        return False
    return False


# =========================================================
# Schemas
# =========================================================
class DashboardStats(BaseModel):
    total_documents: int
    total_chats: int
    active_users: int
    avg_response_time: str


class DocumentOut(BaseModel):
    id: int
    name: str
    category: Optional[str] = None
    size_mb: Optional[float] = None
    uploaded_at: Optional[str] = None
    is_active: bool = True
    chunks_count: int = 0


class ToggleOut(BaseModel):
    ok: bool
    id: int
    is_active: bool


# ---------- Knowledge Schemas ----------
class KnowledgeOut(BaseModel):
    id: int
    title: str
    content: str
    category: str
    created_at: Optional[str] = None
    is_active: bool


class KnowledgeCreate(BaseModel):
    title: str
    content: str
    category: str
    is_active: Optional[bool] = None


class KnowledgeUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    category: Optional[str] = None
    is_active: Optional[bool] = None


class SettingsOut(BaseModel):
    chatbot_available: bool
    system_message: str
    model_status: str
    last_update: Optional[str] = None


class SettingsSave(BaseModel):
    chatbot_available: Optional[bool] = None
    system_message: Optional[str] = None


# =========================================================
# Dashboard
# =========================================================
@router.get("/stats", response_model=DashboardStats)
def admin_dashboard_stats(
    db: Session = Depends(get_db),
    _admin: AdminUser = Depends(verify_admin),
):
    total_documents = safe_count(db, Document)

    # ✅ تم إلغاء تخزين المحادثات في قاعدة البيانات
    total_chats = 0
    active_users = 0

    return DashboardStats(
        total_documents=total_documents,
        total_chats=total_chats,
        active_users=active_users,
        avg_response_time="—",
    )


# =========================================================
# Documents
# =========================================================
@router.get("/documents", response_model=List[DocumentOut])
def list_documents(
    q: Optional[str] = Query(default=None),
    category: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
    _admin: AdminUser = Depends(verify_admin),
):
    query = db.query(Document)
    if q:
        query = query.filter(Document.name.contains(q))
    if category:
        query = query.filter(Document.category == category)

    docs = query.order_by(desc(Document.upload_date)).all()

    # ✅ عدد الـ chunks لكل وثيقة
    doc_ids = [d.id for d in docs]
    chunks_map: Dict[int, int] = {}
    if doc_ids:
        rows = (
            db.query(DocumentChunk.document_id, func.count(DocumentChunk.id))
            .filter(DocumentChunk.document_id.in_(doc_ids))
            .group_by(DocumentChunk.document_id)
            .all()
        )
        chunks_map = {doc_id: int(cnt) for doc_id, cnt in rows}

    out: List[DocumentOut] = []
    for d in docs:
        out.append(
            DocumentOut(
                id=d.id,
                name=d.name,
                category=d.category,
                size_mb=parse_size_to_mb(d.size),
                uploaded_at=d.upload_date.isoformat() if d.upload_date else None,
                is_active=doc_status_to_bool(d.status),
                chunks_count=chunks_map.get(d.id, 0),
            )
        )
    return out


@router.post("/documents/{doc_id}/toggle", response_model=ToggleOut)
def toggle_document(
    doc_id: int,
    db: Session = Depends(get_db),
    _admin: AdminUser = Depends(verify_admin),
):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(404, "Document not found")

    new_active = not doc_status_to_bool(doc.status)
    doc.status = bool_to_doc_status(new_active)
    db.commit()
    return ToggleOut(ok=True, id=doc_id, is_active=new_active)


@router.get("/documents/{doc_id}/chunks-preview")
def document_chunks_preview(
    doc_id: int,
    limit: int = Query(default=5, ge=1, le=20),
    db: Session = Depends(get_db),
    _admin: AdminUser = Depends(verify_admin),
):
    """✅ يعرض عينة من الـ chunks عشان تتأكد إن الملف تفهرس صح"""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(404, "Document not found")

    chunks = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == doc_id)
        .order_by(desc(DocumentChunk.id))
        .limit(limit)
        .all()
    )

    return {
        "doc_id": doc_id,
        "doc_name": doc.name,
        "chunks_found": len(chunks),
        "chunks": [
            {
                "id": c.id,
                "text_preview": (
                    (getattr(c, "text", None) or getattr(c, "content", None) or "")[:700]
                ),
            }
            for c in chunks
        ],
    }


@router.delete("/documents/{doc_id}")
def delete_document(
    doc_id: int,
    db: Session = Depends(get_db),
    _admin: AdminUser = Depends(verify_admin),
):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(404, "Document not found")

    # ✅ حذف chunks من DB
    db.query(DocumentChunk).filter(DocumentChunk.document_id == doc_id).delete(
        synchronize_session=False
    )

    # ✅ حذف الملف من الهارد لو عندك file_path
    removed_file = safe_remove_file(getattr(doc, "file_path", None))

    # ✅ حذف سجل الوثيقة
    db.delete(doc)
    db.commit()

    return {"ok": True, "id": doc_id, "file_deleted": removed_file}


# =========================================================
# Knowledge
# =========================================================
@router.get("/knowledge", response_model=List[KnowledgeOut])
def list_knowledge(
    q: Optional[str] = Query(default=None),
    category: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
    _admin: AdminUser = Depends(verify_admin),
):
    query = db.query(KnowledgeEntry)
    if q:
        query = query.filter(
            (KnowledgeEntry.title.contains(q)) | (KnowledgeEntry.content.contains(q))
        )
    if category:
        query = query.filter(KnowledgeEntry.category == category)

    rows = query.order_by(desc(KnowledgeEntry.created_at)).all()
    return [
        KnowledgeOut(
            id=r.id,
            title=r.title,
            content=r.content,
            category=r.category,
            created_at=r.created_at.isoformat() if r.created_at else None,
            is_active=bool(r.enabled),
        )
        for r in rows
    ]


@router.post("/knowledge", response_model=KnowledgeOut)
def create_knowledge(
    payload: KnowledgeCreate,
    db: Session = Depends(get_db),
    _admin: AdminUser = Depends(verify_admin),
):
    entry = KnowledgeEntry(
        title=payload.title,
        content=payload.content,
        category=payload.category,
        enabled=bool(payload.is_active) if payload.is_active is not None else True,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)

    return KnowledgeOut(
        id=entry.id,
        title=entry.title,
        content=entry.content,
        category=entry.category,
        created_at=entry.created_at.isoformat() if entry.created_at else None,
        is_active=bool(entry.enabled),
    )


@router.put("/knowledge/{entry_id}", response_model=KnowledgeOut)
def update_knowledge(
    entry_id: int,
    payload: KnowledgeUpdate,
    db: Session = Depends(get_db),
    _admin: AdminUser = Depends(verify_admin),
):
    entry = db.query(KnowledgeEntry).filter(KnowledgeEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(404, "Knowledge entry not found")

    if payload.title is not None:
        entry.title = payload.title
    if payload.content is not None:
        entry.content = payload.content
    if payload.category is not None:
        entry.category = payload.category
    if payload.is_active is not None:
        entry.enabled = bool(payload.is_active)

    entry.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(entry)

    return KnowledgeOut(
        id=entry.id,
        title=entry.title,
        content=entry.content,
        category=entry.category,
        created_at=entry.created_at.isoformat() if entry.created_at else None,
        is_active=bool(entry.enabled),
    )


@router.post("/knowledge/{entry_id}/toggle")
def toggle_knowledge(
    entry_id: int,
    db: Session = Depends(get_db),
    _admin: AdminUser = Depends(verify_admin),
):
    entry = db.query(KnowledgeEntry).filter(KnowledgeEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(404, "Knowledge entry not found")

    entry.enabled = not bool(entry.enabled)
    entry.updated_at = datetime.utcnow()
    db.commit()
    return {"ok": True, "id": entry_id, "is_active": bool(entry.enabled)}


@router.delete("/knowledge/{entry_id}")
def delete_knowledge(
    entry_id: int,
    db: Session = Depends(get_db),
    _admin: AdminUser = Depends(verify_admin),
):
    entry = db.query(KnowledgeEntry).filter(KnowledgeEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(404, "Knowledge entry not found")

    db.delete(entry)
    db.commit()
    return {"ok": True, "id": entry_id}


# =========================================================
# Settings
# =========================================================
@router.get("/settings", response_model=SettingsOut)
def get_settings(
    db: Session = Depends(get_db),
    _admin: AdminUser = Depends(verify_admin),
):
    s = db.query(SystemSettings).first()
    if not s:
        return SettingsOut(
            chatbot_available=True,
            system_message="أنت مساعد ذكي مفيد ومهذب.",
            model_status="نشط",
            last_update=None,
        )

    return SettingsOut(
        chatbot_available=bool(s.chatbot_available),
        system_message=s.system_message or "",
        model_status=s.model_status or "—",
        last_update=s.last_update.isoformat() if s.last_update else None,
    )


@router.post("/settings", response_model=Dict[str, Any])
def save_settings(
    payload: SettingsSave,
    db: Session = Depends(get_db),
    _admin: AdminUser = Depends(verify_admin),
):
    s = db.query(SystemSettings).first()
    if not s:
        s = SystemSettings()
        db.add(s)
        db.commit()
        db.refresh(s)

    if payload.chatbot_available is not None:
        s.chatbot_available = bool(payload.chatbot_available)
    if payload.system_message is not None:
        s.system_message = payload.system_message

    s.last_update = datetime.utcnow()
    if not s.model_status:
        s.model_status = "نشط"

    db.commit()
    return {"ok": True}
