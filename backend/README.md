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

# LLM Configuration
LLM_MODEL=gpt-3.5-turbo
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
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

- `POST /api/chat` - Send a message and get AI response
- `GET /api/chats` - Get all chats for a user role
- `GET /api/chats/{chat_id}/messages` - Get messages in a chat
- `POST /api/feedback` - Submit feedback for a chat

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

Example API call:

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

### LLM Issues
- Verify OpenAI API key is set correctly
- Check API quota and rate limits
- System will fallback to basic responses if API unavailable

### Document Processing Issues
- Ensure required libraries are installed (pypdf, docx2python)
- Check file format support
- Verify file is not corrupted

## License

This project is part of the chatbot application.

