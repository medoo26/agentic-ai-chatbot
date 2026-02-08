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

from database import get_db, Document, DocumentChunk, AdminUser

security = HTTPBasic()
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

router = APIRouter(prefix="/api/admin", tags=["admin"])


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


class DocumentOut(BaseModel):
    id: int
    public_id: Optional[str] = None
    name: str
    size_mb: Optional[float] = None
    uploaded_at: Optional[str] = None
    is_active: bool = True


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
