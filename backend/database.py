# database.py
from __future__ import annotations

import os
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
)
from sqlalchemy.orm import sessionmaker, relationship, declarative_base

load_dotenv()

# ---------------------------------------------------------
# Database Config
# ---------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chatbot.db").strip()

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ---------------------------------------------------------
# RAG Tables
# ---------------------------------------------------------
class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
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


Index("ix_document_chunks_document_id", DocumentChunk.document_id)

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


# ---------------------------------------------------------
# Admin / System Tables
# ---------------------------------------------------------
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
# Init / DB Session
# ---------------------------------------------------------
def init_db():
    """إنشاء الجداول (لن يحذف أي جداول قديمة موجودة)"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """توفير جلسة اتصال لقاعدة البيانات"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
