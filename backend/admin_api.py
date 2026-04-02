# admin_api.py
from __future__ import annotations

import os
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import desc

from passlib.context import CryptContext

from database import (
    get_db,
    Document,
    DocumentChunk,
    AdminUser,
    SystemSettings,
    KnowledgeEntry,
)

security = HTTPBasic()
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

router = APIRouter(prefix="/api/admin", tags=["admin"])


# =========================================================
# Auth
# =========================================================
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
        return round(num, 2)
    except Exception:
        return None


def safe_remove_file(path: Optional[str]) -> bool:
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
class DocumentOut(BaseModel):
    id: int
    public_id: Optional[str] = None
    name: str
    size_mb: Optional[float] = None
    uploaded_at: Optional[str] = None
    is_active: bool = True


class AdminStatsOut(BaseModel):
    total_documents: int
    total_chunks: int
    total_knowledge_entries: int
    active_knowledge_entries: int
    chatbot_available: bool
    last_upload_at: Optional[str] = None


class BulkDeleteRequest(BaseModel):
    ids: List[int]


# =========================================================
# Routes
# =========================================================
@router.get("/stats", response_model=AdminStatsOut)
def admin_stats(
    db: Session = Depends(get_db),
    _admin: AdminUser = Depends(verify_admin),
):
    total_docs = db.query(Document).count()
    total_chunks = db.query(DocumentChunk).count()

    total_ke = db.query(KnowledgeEntry).count()
    active_ke = db.query(KnowledgeEntry).filter(KnowledgeEntry.enabled == True).count()

    settings = db.query(SystemSettings).order_by(desc(SystemSettings.id)).first()
    chatbot_available = bool(settings.chatbot_available) if settings else True

    last_doc = db.query(Document).order_by(desc(Document.upload_date)).first()
    last_upload = (
        last_doc.upload_date.isoformat()
        if last_doc and last_doc.upload_date
        else None
    )

    return AdminStatsOut(
        total_documents=total_docs,
        total_chunks=total_chunks,
        total_knowledge_entries=total_ke,
        active_knowledge_entries=active_ke,
        chatbot_available=chatbot_available,
        last_upload_at=last_upload,
    )


@router.get("/documents", response_model=List[DocumentOut])
def list_documents(
    q: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
    _admin: AdminUser = Depends(verify_admin),
):
    query = db.query(Document)
    if q:
        query = query.filter(Document.name.contains(q))

    docs = query.order_by(desc(Document.upload_date)).all()

    return [
        DocumentOut(
            id=d.id,
            public_id=getattr(d, "public_id", None),
            name=d.name,
            size_mb=parse_size_to_mb(d.size),
            uploaded_at=d.upload_date.isoformat() if d.upload_date else None,
            is_active=True,
        )
        for d in docs
    ]


@router.delete("/documents/{doc_id}")
def delete_document(
    doc_id: int,
    db: Session = Depends(get_db),
    _admin: AdminUser = Depends(verify_admin),
):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(404, "Document not found")

    db.query(DocumentChunk).filter(DocumentChunk.document_id == doc_id).delete(
        synchronize_session=False
    )

    removed_file = safe_remove_file(getattr(doc, "file_path", None))

    db.delete(doc)
    db.commit()

    return {"ok": True, "id": doc_id, "file_deleted": removed_file}


@router.post("/documents/bulk-delete")
def bulk_delete_documents(
    payload: BulkDeleteRequest,
    db: Session = Depends(get_db),
    _admin: AdminUser = Depends(verify_admin),
):
    ids = list(set(payload.ids or []))
    if not ids:
        raise HTTPException(status_code=400, detail="لم يتم إرسال أي ملفات للحذف.")

    docs = db.query(Document).filter(Document.id.in_(ids)).all()
    if not docs:
        raise HTTPException(status_code=404, detail="لم يتم العثور على الملفات المحددة.")

    deleted_ids: List[int] = []
    file_deleted_count = 0

    try:
        for doc in docs:
            db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).delete(
                synchronize_session=False
            )

            removed_file = safe_remove_file(getattr(doc, "file_path", None))
            if removed_file:
                file_deleted_count += 1

            deleted_ids.append(doc.id)
            db.delete(doc)

        db.commit()

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"فشل حذف الملفات: {str(e)}")

    return {
        "ok": True,
        "deleted_count": len(deleted_ids),
        "deleted_ids": deleted_ids,
        "file_deleted_count": file_deleted_count,
    }