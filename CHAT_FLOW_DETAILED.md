# Chatbot Flow: From User Query to Answer (Detailed)

This document describes **step-by-step** how a user message travels from the UI to the backend and back until the user sees the assistant’s reply.

---

## High-Level Overview

```
User types message → ChatPage (React) → apiPost("/api/chat") → FastAPI main.py
  → LLM refine_query → (optional: file request) OR (RAG search → LLM generate_response)
  → MessageResponse → Frontend → User sees answer
```

---

## 1. Frontend: User Input and Send

**File:** `src/app/pages/ChatPage.tsx`

1. User types in the input and submits (Enter or Send button).
2. **`sendMessage(content)`** runs:
   - Validates: non-empty content and not already sending.
   - Sets **`sending = true`** (disables input/button).
   - Builds a **user message** object (`UIMessage`) with:
     - `id`, `content`, `sender: "user"`, `timestamp`, `sources`, `attachments`.
   - **Appends the user message** to `messages` and clears the input.
   - Calls **`apiPost<ApiMessage>("/api/chat", { content })`** (from `src/api.ts`).
   - On success: builds an **assistant message** from the API response (`content`, `sources`, `attachments`) and appends it to `messages`.
   - On error: appends an error message as if from the assistant.
   - In **`finally`**: sets **`sending = false`**.

**File:** `src/api.ts`

3. **`apiPost("/api/chat", { content })`**:
   - Sends **POST** to `API_BASE_URL + "/api/chat"` (default `http://127.0.0.1:8000/api/chat`).
   - Headers: `Accept: application/json`, `Content-Type: application/json`.
   - Body: `JSON.stringify({ content })`.
   - Uses **`handleResponse(res)`**: reads body as text, parses JSON; if `!res.ok`, throws with `detail` or `message`.
   - Returns the parsed JSON (the chat API response).

So: **user query** is sent as **`{ "content": "user text" }`** to the backend.

---

## 2. Backend: Entry Point and Validation

**File:** `backend/main.py`

4. **`POST /api/chat`** is handled by **`send_message(message: MessageCreate, db: Session)`**.
5. **`user_text = (message.content or "").strip()`**. If empty → **400** "اكتب سؤالك من فضلك."
6. **Optional: “choose doc” reply**
   - If the last bot turn asked “أي ملف تقصد؟” (`LAST_TOPIC.endswith("::choose_doc")`) and the user replied with a number (1–10) or a document name:
   - Backend resolves the chosen document and sets **`forced_doc_key`** (and updates `PENDING_CHOICES` / `LAST_TOPIC`). This is used later in RAG to restrict search to that document.

---

## 3. Query Refinement (Understand and Classify the Query)

**File:** `backend/main.py` → **`llm_service.refine_query(...)`**  
**File:** `backend/llm_service.py` → **`refine_query()`**, **`_handle_greeting()`**, **`_gemini_refine_json()`** / **`_local_fallback_refine()`**

7. **`refined = llm_service.refine_query(user_text, context_hint="PSAU University chatbot...")`**
   - **Greetings:** If the text matches patterns (سلام، أهلاً، كيف حالك، شكراً، etc.), **`_handle_greeting()`** returns a fixed reply. Then `refine_query` returns a dict that includes **`direct_response`**.
   - **Otherwise:** The refiner (e.g. Gemini) is called to get a **JSON** with:
     - **`refined_question`**: clear Arabic formulation of the user intent.
     - **`intent`**: e.g. `academic_procedure`, `admission`, `schedules`, `tuition`, `general_info`, `greetings`, etc.
     - **`entities`**, **`constraints`**, **`search_queries`**: for search/RAG.
     - **`needs_clarification`**, **`clarifying_question`**.
     - **`request_type`**: **`"answer"`** or **`"file"`** (file = user wants a document/file/download).
     - **`file_query`**: short description of the file when **`request_type == "file"`**.
   - If the refiner fails, **`_local_fallback_refine()`** is used (keyword-based, e.g. نموذج، ملف، تحميل → **`request_type: "file"`**).

8. **Direct response (no RAG, no file fetch)**  
   If **`refined.get("direct_response")`** is set (e.g. greeting):
   - Backend immediately returns **`MessageResponse`** with that text, **sources=[]**, **attachments=[]**.  
   - Frontend then appends this as the assistant message. **Flow ends here for greetings.**

---

## 4. File Request Branch (User Wants a Document)

**File:** `backend/main.py` (inside **`send_message`**)

9. If **`request_type == "file"`**:
   - **`file_query`** is taken from refined (or fallback to **`user_text`** if too short).
   - **`_find_best_document(db, file_query)`**:
     - Normalizes query (Arabic, digits), extracts keywords, optionally matches by file extension.
     - Queries **`Document`** (DB) by name/keywords and returns the best match.
   - If **no document** found → return a message like “لم أجد ملفًا مطابقًا...”.
   - If **document** found:
     - Ensures it has **`public_id`** (for download URL).
     - Builds **`AttachmentOut`** with **`name`**, **`url="/api/files/{public_id}"`**, **`mime`**.
     - Returns **`MessageResponse`** with a short text (e.g. “تفضل، هذا الملف المطلوب: **{name}**”) and **attachments=[attachment]**.
   - Frontend shows the message and a download link using **`/api/files/{public_id}`**. **Flow ends here for file requests.**

---

## 5. Normal Answer Branch: RAG (Retrieval + Answer)

When **`request_type != "file"`** and there is **no** **`direct_response`**, the backend does **RAG**: build a retrieval query → search vector store → get context → generate answer.

### 5.1 Build Retrieval Query and Topic Memory

**File:** `backend/main.py`

10. **`refined_question`**, **`intent`** from **`refined`**.
11. **`retrieval_query = llm_service.build_retrieval_query_smart(refined, user_text)`**  
    - Uses **`refined_question`** / **`search_queries`** and, if the user asked for a specific **article** (e.g. “المادة 4”), appends article-related phrasing (e.g. “نص المادة” or “نص القواعد التنفيذية”) so RAG can retrieve the right section.
12. **Follow-up handling:**  
    If the user message is short, referential (e.g. “اهدافها”, “مهامها”), and **`LAST_TOPIC`** is set (and not “choose_doc”), **`retrieval_query`** is combined with **`LAST_TOPIC`** so the system “remembers” what was being discussed.
13. **Article phrase:**  
    If the user mentioned a specific article (e.g. “المادة الخامسة”), **`_extract_article_phrase(user_text)`** and **`_is_exec_rules_query(user_text)`** are used to optionally add “القواعد التنفيذية” or “نص المادة” to **`retrieval_query`**.
14. **`LAST_TOPIC`** is updated from **`refined_question`** when it’s long enough (≥20 chars), for the next turn.

### 5.2 Retrieve Context (RAG Search)

**File:** `backend/main.py` (broad search, doc selection, focused search)  
**File:** `backend/rag_system.py` (**RAGSystem**: ChromaDB, embeddings, **search** / **search_smart** / **search_article_exact**)

15. **Config:**  
    **`BROAD_TOP_K`**, **`FOCUSED_TOP_K`**, **`NEIGHBOR_WINDOW`**, **`DIST_THRESHOLD`** from env (with defaults).
16. **Two main paths:**

    **A) Article-specific query (e.g. “المادة 4”):**
    - **`rag_system.search_smart(retrieval_query, ...)`** is called first:
      - In **`rag_system.py`**, **`search_smart`** detects article number from the query.
      - If an article is detected, it tries **`search_article_exact(...)`** (metadata match on **article_no** / **article_no_norm**); if that returns results, those are used.
      - Otherwise it falls back to normal **semantic search** with optional **where** (e.g. **doc_key**).
    - Results are filtered by **`_filter_hits_for_article(hits, article_phrase)`** so only chunks that actually contain that article are kept.
    - If **no context** from **search_smart**:
      - **Broad search** (no **doc_key**): **`rag_system.search(retrieval_query, top_k=BROAD_TOP_K, ...)`**.
      - If multiple documents match the article, the backend may ask “أي ملف تقصد؟” and set **`LAST_TOPIC = "…::choose_doc"`** and **`PENDING_CHOICES`**; then returns that message and **stops** (user must choose next).
      - Otherwise: **`_pick_best_doc_key_from_hits(...)`** or **`forced_doc_key`** → **`best_doc_key`**.
      - **`_retrieve_article_context(...)`** is called (3-layer strategy: exact article metadata → semantic + **where** → legacy **h2**/article_high_level) to get **`context_docs`**.

    **B) Non-article query:**
    - **Broad search:** **`rag_system.search(retrieval_query, top_k=BROAD_TOP_K, ...)`** → **`broad_hits`**.
    - **`best_doc_key = forced_doc_key2 or _pick_best_doc_key_from_hits(broad_hits)`**.
    - **Focused search:** **`rag_system.search(retrieval_query, top_k=FOCUSED_TOP_K, include_neighbors=True, neighbor_window=NEIGHBOR_WINDOW, where={"doc_key": best_doc_key})`** → **`context_docs`**.

17. **Fallback:** If **`context_docs`** is still empty but **`broad_hits`** exists, another **`rag_system.search(...)`** is run **without** **where** to get some context.
18. **Distance threshold:** For **non-article** queries, if the best score is **worse** (larger) than **`DIST_THRESHOLD`**, **`context_docs`** can be cleared to avoid irrelevant context.
19. **Result:** **`context_docs`** = list of **{ "content", "metadata", "score" }** chunks from ChromaDB (and possibly **`sources`** list for citation).

### 5.3 Generate the Answer

**File:** `backend/main.py` → **`llm_service.generate_response(...)`**  
**File:** `backend/llm_service.py` → **`generate_response()`**, **`_build_context()`**, **`_extract_*`**, **`append_sources_to_answer()`**

20. **`ai_text = llm_service.generate_response(user_query=user_text, refined_question=refined_question, intent=intent, context_docs=context_docs, ...)`**
    - **Greeting:** If the user message is again detected as greeting, returns the greeting text (no LLM).
    - **Exec rules for one article:** If the request is “القواعد التنفيذية للمادة X”, **`_extract_exec_rules_only(...)`** may extract that block from **context_docs** and return it (with optional sources).
    - **Single-article request:** If the user asked for “نص المادة X”, **`_extract_requested_article_only(...)`** may extract that article from context and return it (with optional sources).
    - **Table-only:** If the best doc is a table (metadata or content heuristics), **`_should_return_table_directly(...)`** can return the table text (with optional sources).
    - **Otherwise:**
      - **`_build_context(context_docs)`** builds a single string from the best chunks (with source labels and length limits).
      - If **no context** → return “عذراً، هذه المعلومة غير متوفرة في الملفات المرفوعة حالياً.”
      - System + user messages are built (with optional article-specific instructions).
      - **OpenAI LLM** is invoked with these messages. On failure (e.g. 429), retry once; if still failing, **Gemini** is used as fallback (**`_gemini_answer_from_context`**).
      - **`append_sources_to_answer(answer, context_docs)`** may append “المصادر:” and source names to the answer (if configured).

21. **`sources = llm_service.extract_sources(context_docs, max_sources=...)`**  
    - List of document names (from metadata) to attach to the response.

22. **Return**  
    **`MessageResponse(id=..., content=ai_text, sender="assistant", timestamp=..., sources=sources, attachments=[])`**.

---

## 6. Back to the Frontend

**File:** `src/app/pages/ChatPage.tsx`

23. **`apiPost("/api/chat", { content })`** resolves with the **`MessageResponse`** object.
24. **`sendMessage`** builds **`aiMsg`** from:
   - **`content`** → message body (may be Markdown).
   - **`sources`** → shown under the bubble as “المصادر”.
   - **`attachments`** → shown as “ملفات للتحميل” with **تحميل** linking to **`API_BASE_URL + url`** (e.g. **`/api/files/{public_id}`**).
25. **`setMessages(prev => [...prev, aiMsg])** adds the assistant message to the list.
26. **React** re-renders; **ScrollArea** shows the new message; **ReactMarkdown** + **remarkGfm** render **content** (tables, lists, etc.).
27. **`sending = false`** (in **finally**), so the user can type again.

---

## 7. Summary Diagram (Backend)

```
User "ما هي المادة الخامسة؟"
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  POST /api/chat { content }                                    │
│  → send_message()                                              │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  refine_query(user_text)                                       │
│  → Greeting? → direct_response → return MessageResponse        │
│  → request_type "file"? → _find_best_document → AttachmentOut  │
│     → return MessageResponse with attachments                  │
└───────────────────────────────────────────────────────────────┘
        │
        ▼ (request_type == "answer")
┌───────────────────────────────────────────────────────────────┐
│  build_retrieval_query_smart() + article phrase + LAST_TOPIC    │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  RAG: search_smart or search (broad → best_doc_key → focused)  │
│  → context_docs (ChromaDB chunks)                              │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  generate_response(user_query, refined_question, intent,        │
│                    context_docs)                               │
│  → exec rules / single article / table direct OR               │
│  → _build_context() → OpenAI (or Gemini fallback) → ai_text    │
│  → extract_sources()                                           │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
  MessageResponse(content=ai_text, sources=sources, attachments=[])
        │
        ▼
  Frontend: setMessages([...prev, aiMsg]) → user sees answer
```

---

## 8. Key Files Reference

| Step              | Location |
|-------------------|----------|
| User input & send | `src/app/pages/ChatPage.tsx` → `sendMessage()` |
| HTTP client       | `src/api.ts` → `apiPost()` |
| Chat endpoint     | `backend/main.py` → `send_message()` |
| Refine query      | `backend/llm_service.py` → `refine_query()` |
| File lookup       | `backend/main.py` → `_find_best_document()` |
| RAG search        | `backend/rag_system.py` → `search()`, `search_smart()`, `search_article_exact()` |
| Article retrieval | `backend/main.py` → `_retrieve_article_context()` |
| Answer generation | `backend/llm_service.py` → `generate_response()` |
| Response format   | `backend/main.py` → `MessageResponse` |

This is the full process from **user query** to **user gets answer** in your chatbot.
