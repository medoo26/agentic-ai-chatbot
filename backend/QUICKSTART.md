# Quick Start Guide

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Set Up Environment

Copy the example environment file:

```bash
# On Windows
copy env.example .env

# On Linux/Mac
cp env.example .env
```

Edit `.env` and add your OpenAI API key (optional but recommended):

```env
OPENAI_API_KEY=sk-your-actual-api-key-here
```

## 3. Initialize Database

```bash
python setup.py
```

This will:
- Create the database
- Create default admin user (username: `admin`, password: `admin123`)

## 4. Start the Server

```bash
python run.py
```

Or:

```bash
uvicorn main:app --reload
```

## 5. Test the API

Visit `http://localhost:8000/docs` for interactive API documentation.

## Testing Endpoints

### Test Chat Endpoint

```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"content": "مرحبا", "chat_id": null}'
```

### Test Admin Endpoint (with authentication)

```bash
curl -X GET "http://localhost:8000/api/admin/documents" \
  -u admin:admin123
```

## Frontend Integration

Update your frontend to use the backend API. In your frontend code, set the API base URL:

```typescript
const API_BASE_URL = 'http://localhost:8000';
```

## Troubleshooting

### Import Errors
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Database Errors
Delete `chatbot.db` and run `python setup.py` again.

### Vector Store Errors
Delete the `vector_store` directory and restart the server.

### OpenAI API Errors
If you don't have an API key, the system will use local embeddings but LLM responses will be limited. You can still test the RAG system with local models.

