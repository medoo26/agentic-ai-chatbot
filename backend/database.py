# database.py
from __future__ import annotations

import os
import uuid
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Boolean,
    ForeignKey,
    Index,
    text,
)
from sqlalchemy.orm import sessionmaker, relationship, declarative_base

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chatbot.db").strip()

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ---------------------------------------------------------
# Models
# ---------------------------------------------------------
class Document(Base):
    __tablename__ = "documents"
    # ✅ يمنع إعادة استخدام id في SQLite (AUTOINCREMENT)
    __table_args__ = {"sqlite_autoincrement": True}

    id = Column(Integer, primary_key=True, index=True)

    # ✅ معرف ثابت للتحميل (لا يتغير ولا يتكرر)
    public_id = Column(String, unique=True, index=True, nullable=True)

    name = Column(String, nullable=False)
    category = Column(String, nullable=False)
    file_path = Column(String, nullable=False)

    status = Column(String, default="نشط")
    size = Column(String)
    upload_date = Column(DateTime, default=datetime.utcnow)

    chunks = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)

    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    embedding_id = Column(String)

    document = relationship("Document", back_populates="chunks")


class KnowledgeEntry(Base):
    __tablename__ = "knowledge_entries"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    category = Column(String, nullable=False)
    enabled = Column(Boolean, default=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    embedding_id = Column(String)


class SystemSettings(Base):
    __tablename__ = "system_settings"

    id = Column(Integer, primary_key=True, index=True)
    chatbot_available = Column(Boolean, default=True)
    system_message = Column(
        Text, default="أنت مساعد ذكي مفيد ومهذب للرد على استفسارات الجامعة."
    )
    model_status = Column(String, default="نشط")
    last_update = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AdminUser(Base):
    __tablename__ = "admin_users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# ---------------------------------------------------------
# Init + Auto-migration for public_id
# ---------------------------------------------------------
def init_db():
    """ينشئ الجداول لو غير موجودة"""
    Base.metadata.create_all(bind=engine)
    ensure_documents_public_id_column()


def ensure_documents_public_id_column():
    """
    ✅ حل دائم:
    - يضيف عمود public_id لو القاعدة قديمة وما فيها العمود.
    - يعبّي public_id للصفوف القديمة.
    """
    if "sqlite" not in DATABASE_URL.lower():
        # لو انت مستخدم DB ثانية، اعمل migration بالطريقة المناسبة
        return

    with engine.begin() as conn:
        # هل العمود موجود؟
        cols = conn.execute(text("PRAGMA table_info(documents)")).fetchall()
        col_names = {c[1] for c in cols}  # c[1] = name
        if "public_id" not in col_names:
            conn.execute(text("ALTER TABLE documents ADD COLUMN public_id TEXT"))
            conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_documents_public_id ON documents(public_id)"))

        # عبّي القيم الناقصة
        rows = conn.execute(
            text("SELECT id FROM documents WHERE public_id IS NULL OR public_id = ''")
        ).fetchall()

        for (doc_id,) in rows:
            pid = uuid.uuid4().hex
            conn.execute(
                text("UPDATE documents SET public_id = :pid WHERE id = :id"),
                {"pid": pid, "id": doc_id},
            )


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
