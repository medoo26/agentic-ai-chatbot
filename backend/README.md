# Chatbot Backend API

A Python backend for the chatbot application with RAG (Retrieval-Augmented Generation), LLM integration, and SQL database.

## Features

- **RAG System**: Retrieval-Augmented Generation using ChromaDB vector store
- **LLM Integration**: OpenAI GPT models for generating responses
- **SQL Database**: SQLite/PostgreSQL for storing chats, messages, documents, and knowledge base
- **Document Processing**: Support for PDF, DOCX, and TXT files
- **Admin Panel**: Full CRUD operations for documents, knowledge base, and chat logs
- **Authentication**: Basic authentication for admin endpoints

## Setup

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the `backend` directory:

```env
# Database
DATABASE_URL=sqlite:///./chatbot.db

# OpenAI API (for LLM)
OPENAI_API_KEY=your_openai_api_key_here

# Admin Credentials
ADMIN_USERNAME=admin
ADMIN_PASSWORD=admin123

# Server Configuration
HOST=0.0.0.0
PORT=8000

# CORS
CORS_ORIGINS=http://localhost:5173,http://localhost:3000

# Vector Store
VECTOR_STORE_PATH=./vector_store
CHROMA_COLLECTION=university_docs_legal_v2

# LLM Configuration
LLM_MODEL=gpt-3.5-turbo
LOCAL_EMBEDDING_MODEL=intfloat/multilingual-e5-large
```

**Note**: If you don't have an OpenAI API key, the system will use local sentence transformers for embeddings, but LLM responses will be limited.

### 3. Initialize Database

The database will be automatically created on first run. To manually initialize:

```python
from database import init_db
init_db()
```

### 4. Run the Server

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Chat Endpoints

- `POST /api/chat` - Send a message and get a full JSON response (non-streaming fallback)
- `POST /api/chat/stream` - Send a message and receive SSE streaming events (`token`, `meta`, `done`, `error`)

### Admin Endpoints (Require Authentication)

#### Documents
- `GET /api/admin/documents` - Get all documents
- `POST /api/admin/documents` - Upload a document
- `DELETE /api/admin/documents/{document_id}` - Delete a document

#### Knowledge Base
- `GET /api/admin/knowledge` - Get all knowledge entries
- `POST /api/admin/knowledge` - Create a knowledge entry
- `PUT /api/admin/knowledge/{entry_id}` - Update a knowledge entry
- `DELETE /api/admin/knowledge/{entry_id}` - Delete a knowledge entry

#### Chat Logs
- `GET /api/admin/logs` - Get chat logs

#### Dashboard
- `GET /api/admin/stats` - Get dashboard statistics

#### Settings
- `GET /api/admin/settings` - Get system settings
- `PUT /api/admin/settings` - Update system settings

### Health Check

- `GET /api/health` - Check API health and service availability

## Database Schema

### Tables

- **users**: User information
- **chats**: Chat sessions
- **messages**: Chat messages
- **feedback**: User feedback
- **documents**: Uploaded documents
- **document_chunks**: Document chunks for vector store
- **knowledge_entries**: Knowledge base entries
- **chat_logs**: Logged chat interactions
- **system_settings**: System configuration
- **admin_users**: Admin user accounts

## RAG System

The RAG system uses:
- **ChromaDB** for vector storage
- **Sentence Transformers** or **OpenAI Embeddings** for text embeddings
- **LangChain** for document processing and chunking

Documents and knowledge entries are automatically:
1. Split into chunks
2. Embedded using the selected embedding model
3. Stored in the vector database
4. Retrieved during query time for context

## LLM Service

The LLM service supports:
- **OpenAI GPT models** (requires API key)
- Fallback responses when API is unavailable

Responses are generated using:
- User query
- Retrieved context from RAG system
- User role
- System message configuration

## Document Processing

Supported formats:
- PDF (`.pdf`)
- Word documents (`.docx`, `.doc`)
- Text files (`.txt`)

Documents are processed to extract text, which is then:
- Cleaned and normalized
- Split into chunks
- Embedded and stored in vector database

## Frontend Integration

Update your frontend API base URL to point to the backend:

```typescript
const API_BASE_URL = 'http://localhost:8000';
```

Example non-streaming API call:

```typescript
const response = await fetch(`${API_BASE_URL}/api/chat`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    content: 'سؤالك هنا',
    chat_id: null, // or existing chat ID
  }),
});
```

Example streaming API call (SSE over `fetch` stream):

```typescript
const sessionId = localStorage.getItem("chatSessionId") || crypto.randomUUID();
localStorage.setItem("chatSessionId", sessionId);

const response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "Accept": "text/event-stream",
    "X-Chat-Session-Id": sessionId,
  },
  body: JSON.stringify({
    content: "سؤالك هنا",
    session_id: sessionId,
  }),
});
```

SSE event contract:
- `event: token` + `{"delta":"..."}`: incremental answer chunks.
- `event: meta` + `{"id","timestamp","sources","attachments","choices","debug"}`: final structured metadata.
- `event: done` + `{"ok":true}`: stream completed.
- `event: error` + `{"message":"..."}`: recoverable stream error.

Session behavior:
- Chat context state is tracked per session id instead of process-global memory.
- Provide the same `X-Chat-Session-Id` (and optional `session_id` body) to preserve follow-up context.
- Use different session ids for independent browser chats/tabs.

For admin endpoints, use Basic Authentication:

```typescript
const credentials = btoa(`${username}:${password}`);
const response = await fetch(`${API_BASE_URL}/api/admin/documents`, {
  headers: {
    'Authorization': `Basic ${credentials}`,
  },
});
```

## Development

### Project Structure

```
backend/
├── main.py                 # FastAPI application
├── database.py             # Database models and setup
├── rag_system.py           # RAG system implementation
├── llm_service.py          # LLM service
├── document_processor.py   # Document processing
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables
└── README.md              # This file
```

### Adding New Features

1. Update database models in `database.py`
2. Add new endpoints in `main.py`
3. Update RAG system if needed in `rag_system.py`
4. Test endpoints using the interactive API docs at `/docs`

## Troubleshooting

### Database Issues
- Ensure SQLite file permissions are correct
- For PostgreSQL, check connection string format

### Vector Store Issues
- Ensure `vector_store` directory has write permissions
- Clear vector store if corrupted: delete `vector_store` directory
- For a full re-index migration, use a new `CHROMA_COLLECTION` value and re-upload/reprocess documents

### LLM Issues
- Verify OpenAI API key is set correctly
- Check API quota and rate limits
- System will fallback to basic responses if API unavailable

### Document Processing Issues
- Ensure required libraries are installed (pypdf, docx2python, python-docx)
- Check file format support
- Verify file is not corrupted

## Legal Indexing v2 (Current Pipeline)

The upload/save flow is unchanged, but indexing is rebuilt:

1. Convert extracted text to structured HTML.
2. Remove repeated page headers/footers and preserve page markers.
3. Normalize HTML tables into canonical readable text.
4. Split strictly by legal hierarchy using HTML headers:
   - `h1 -> level_1`
   - `h2 -> level_2`
   - `h3 -> article`
5. Build rich metadata per chunk (`page_number`, hierarchy path, table metadata).
6. Save vectors + metadata into Chroma (`CHROMA_COLLECTION`).

### Hard Reset / Re-index Guide

When you change indexing schema:

1. Set a new collection name in `.env`:
   - `CHROMA_COLLECTION=university_docs_legal_v2`
2. Restart backend.
3. Re-upload or reprocess all documents so chunks are regenerated in the new collection.
4. Optionally keep the old collection for rollback, or delete old vectors later.

## License

This project is part of the chatbot application.

