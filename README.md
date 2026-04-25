# 🎓 Rukn Al-Aredh – AI Chatbot for University Systems

AI-powered chatbot designed to help students, faculty, and staff quickly access university regulations, documents, and procedures using Retrieval-Augmented Generation (RAG).

---

## 🚀 Overview

This project solves a real problem: difficulty accessing official university information quickly.

The chatbot uses a **RAG pipeline** to:

1. Understand the user’s question
2. Retrieve relevant information from official documents
3. Generate accurate, context-based answers

---

## 🧠 Key Features

* 🔍 Smart search over university regulations
* 📄 Source-grounded answers (reduces hallucination)
* 📑 File attachments & document preview
* ⚡ Fast responses using optimized retrieval
* 🎯 Handles 100+ academic and administrative queries

---

## 🏗️ Tech Stack

### Backend

* Python (FastAPI)
* ChromaDB (Vector Database)
* Sentence Transformers (Embeddings)
* OpenAI / Gemini (LLMs)

### Frontend

* React + TypeScript
* Tailwind CSS
* shadcn/ui components

---

## ⚙️ How It Works

1. Documents are converted into structured HTML
2. Content is split based on headings (H1, H2, H3)
3. Each chunk is embedded and stored in ChromaDB
4. User query is processed and matched with relevant chunks
5. LLM generates a final grounded answer

---

## 📦 Installation

### 1. Clone the repo

```bash
git clone https://github.com/medoo26/agentic-ai-chatbot.git
cd agentic-ai-chatbot
```

### 2. Backend setup

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### 3. Frontend setup

```bash
npm install
npm run dev
```

---

## 🔐 Environment Variables

Create a `.env` file:

```
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

---

## 📌 Future Improvements

* Multi-agent system
* Better document ranking
* Voice interaction
* Mobile support
