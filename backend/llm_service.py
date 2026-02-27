# llm_service.py
import os
import re
import sys
import json
import time
import traceback
from typing import List, Dict, Optional, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Gemini SDK
try:
    import google.generativeai as genai
except Exception:
    genai = None

# -----------------------------------------------------
# Force UTF-8 (Windows)
# -----------------------------------------------------
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

load_dotenv()


class LLMService:
    # -----------------------------------------------------
    # Regex / constants
    # -----------------------------------------------------
    _RE_UUID_PREFIX = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_",
        re.IGNORECASE,
    )

    _RE_TAGS = re.compile(r"<[^>]+>")
    _RE_WS = re.compile(r"\s+")
    _RE_SCRIPT = re.compile(r"<script\b.*?</script>", re.IGNORECASE | re.DOTALL)
    _RE_STYLE = re.compile(r"<style\b.*?</style>", re.IGNORECASE | re.DOTALL)
    _RE_HEAD = re.compile(r"<head\b.*?</head>", re.IGNORECASE | re.DOTALL)

    _RE_CODEFENCE = re.compile(r"```(?:html|xml|json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

    _RE_SALAM = re.compile(
        r"^\s*(السلام\s*عليكم|سلام\s*عليكم|سلام|السلام|السلام عليكم ورحمة الله|السلام عليكم ورحمة الله وبركاته)\s*$",
        re.IGNORECASE,
    )
    _RE_AHLA = re.compile(
        r"^\s*(اهلا|أهلا|اهلين|أهلين|هلا|هلا بك|هلا والله|مرحبا|مرحبًا|يا هلا|حي الله|حيّاك|حيّاك الله)\s*$",
        re.IGNORECASE,
    )
    _RE_HOW_ARE_YOU = re.compile(
        r"^\s*(كيف\s*حال(ك|كم)?|شلون(ك|كم)?|كيفك|كيفكم|طمّني|طمني|وش\s*لونك|وشلونك|علومك)\s*$",
        re.IGNORECASE,
    )
    _RE_THANKS = re.compile(
        r"^\s*(شكرا|شكرًا|يعطيك\s*العافيه|يعطيك\s*العافية|مشكور|تسلم|الله\s*يعافيك|جزاك\s*الله\s*خير|جزاكم\s*الله\s*خير)\s*$",
        re.IGNORECASE,
    )

    _RE_ARTICLE_WORD = re.compile(r"\bالمادة\b", re.IGNORECASE)

    # ✅ طلب "القواعد التنفيذية للمادة ..."
    _RE_EXEC_RULES = re.compile(r"\bالقواعد\s+التنفيذية\s+ل(?:ل)?م(?:اده|ادة)\b", re.IGNORECASE)

    _RE_AR_SPACES = re.compile(r"\s+")
    _RE_TATWEEL = re.compile(r"ـ+")
    _RE_AR_NUM = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

    _RE_HTML_ENT = {
        "&nbsp;": " ",
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&#39;": "'",
    }

    # Markdown-table heuristic
    _RE_MD_TABLE = re.compile(r"^\s*\|?.*?\|.*\n\s*\|?\s*[-:]{3,}", re.MULTILINE)

    # ✅ لالتقاط "المادة ..." من سؤال المستخدم
    _RE_ARTICLE_PHRASE = re.compile(
        r"(?:الماده|المادة)\s+([^\n\.<،,:؛!\?؟]{1,80})",
        re.IGNORECASE,
    )

    def __init__(self):
        # Providers
        self.refiner_provider = (os.getenv("REFINER_PROVIDER") or "gemini").strip().lower()
        self.llm_provider = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()

        # Models
        self.chat_model = (os.getenv("CHAT_MODEL") or "gpt-4o-mini").strip()
        self.refine_model = (os.getenv("REFINE_MODEL") or "models/gemini-flash-latest").strip()
        self.fallback_answer_model = (os.getenv("FALLBACK_ANSWER_MODEL") or "models/gemini-flash-latest").strip()

        # Keys
        self.openai_key = self._extract_openai_key((os.getenv("OPENAI_API_KEY") or "").strip())
        self.gemini_key = (os.getenv("GEMINI_API_KEY") or "").strip()

        # Configs
        self.refine_max_tokens = int(os.getenv("REFINE_MAX_TOKENS") or "256")
        self.refine_temperature = float(os.getenv("REFINE_TEMPERATURE") or "0.1")
        self.answer_temperature = float(os.getenv("ANSWER_TEMPERATURE") or "0.2")

        # Context sizing
        self.max_docs_in_context = int(os.getenv("MAX_DOCS_IN_CONTEXT") or "5")
        self.max_chars_per_doc = int(os.getenv("MAX_CHARS_PER_DOC") or "1400")
        self.max_total_context_chars = int(os.getenv("MAX_TOTAL_CONTEXT_CHARS") or "4500")

        # Article direct output sizing (for "نص المادة")
        self.article_direct_max_chars = int(os.getenv("ARTICLE_DIRECT_MAX_CHARS") or "20000")

        # Sources
        self.append_sources = (os.getenv("APPEND_SOURCES", "0").strip() == "1")
        self.max_sources = int(os.getenv("MAX_SOURCES") or "4")

        # Rate-limit backoff
        self.retry_on_429 = int(os.getenv("RETRY_ON_429") or "1")
        self.retry_sleep_seconds = float(os.getenv("RETRY_SLEEP_SECONDS") or "2.0")

        # HTML conversion configs
        self.html_provider = (os.getenv("HTML_PROVIDER") or "").strip().lower()  # openai|gemini|""
        self.html_model_openai = (os.getenv("HTML_MODEL_OPENAI") or self.chat_model).strip()
        self.html_model_gemini = (os.getenv("HTML_MODEL_GEMINI") or self.fallback_answer_model).strip()
        self.html_max_tokens = int(os.getenv("HTML_MAX_TOKENS") or "2048")
        self.html_temperature = float(os.getenv("HTML_TEMPERATURE") or "0.0")

        # Init OpenAI answer
        self.llm = None
        if self.llm_provider == "openai":
            if not self.openai_key:
                print("Warning: OPENAI_API_KEY is missing or invalid (check .env).")
            else:
                self.llm = self._init_openai_chat(
                    model=self.chat_model,
                    api_key=self.openai_key,
                    temperature=self.answer_temperature,
                )

        # Init Gemini (refine + fallback + html if needed)
        self.gemini_ready = False
        self._init_gemini()

    # -----------------------------------------------------
    # Init helpers
    # -----------------------------------------------------
    def _init_openai_chat(self, model: str, api_key: str, temperature: float):
        configs = [
            dict(model=model, openai_api_key=api_key, temperature=temperature),
            dict(model_name=model, openai_api_key=api_key, temperature=temperature),
            dict(model=model, api_key=api_key, temperature=temperature),
            dict(model_name=model, api_key=api_key, temperature=temperature),
        ]
        last_error = None
        for cfg in configs:
            try:
                llm = ChatOpenAI(**cfg)
                print(f"OpenAI LLM Ready! model={model}")
                return llm
            except Exception as e:
                last_error = e
        print("Failed to initialize OpenAI LLM:", self._safe_str(last_error))
        return None

    def _init_gemini(self):
        if genai is None:
            print("⚠️ google-generativeai not installed. pip install google-generativeai")
            self.gemini_ready = False
            return
        if not self.gemini_key:
            print("⚠️ GEMINI_API_KEY missing (check .env).")
            self.gemini_ready = False
            return
        try:
            genai.configure(api_key=self.gemini_key)
            self.gemini_ready = True
            print(f"Gemini Ready! refine_model={self.refine_model} fallback_answer_model={self.fallback_answer_model}")
        except Exception as e:
            print("⚠️ Gemini init failed:", self._safe_str(e))
            self.gemini_ready = False

    # -----------------------------------------------------
    # Key sanitizer
    # -----------------------------------------------------
    def _extract_openai_key(self, raw: str) -> str:
        if not raw:
            return ""
        raw = raw.strip().strip('"').strip("'")
        if "#" in raw:
            raw = raw.split("#", 1)[0].strip()
        raw = raw.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore").strip()
        m = re.search(r"(sk-[A-Za-z0-9_\-]+)", raw)
        if m:
            return m.group(1).strip()
        cleaned = "".join(ch for ch in raw if ord(ch) < 128).strip()
        return cleaned

    def _safe_str(self, v) -> str:
        if v is None:
            return ""
        try:
            s = v if isinstance(v, str) else str(v)
        except Exception:
            s = repr(v)
        return s.encode("utf-8", errors="replace").decode("utf-8", errors="replace")

    # -----------------------------------------------------
    # Source name cleanup
    # -----------------------------------------------------
    def _clean_source_name(self, name: str) -> str:
        name = (name or "").strip()
        if not name:
            return ""
        name = name.replace("\\", "/")
        if "/" in name:
            name = name.split("/")[-1].strip()
        name = self._RE_UUID_PREFIX.sub("", name).strip()
        name = re.sub(r"\s+", " ", name).strip()
        return name

    # -----------------------------------------------------
    # Arabic normalize
    # -----------------------------------------------------
    def normalize_arabic(self, text: str) -> str:
        if not text:
            return ""
        t = text.strip()
        t = self._RE_TATWEEL.sub("", t)
        t = t.translate(self._RE_AR_NUM)
        t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
        t = t.replace("ى", "ي")
        t = re.sub(r"(.)\1{3,}", r"\1\1", t)
        t = self._RE_AR_SPACES.sub(" ", t).strip()
        return t

    # -----------------------------------------------------
    # Greetings
    # -----------------------------------------------------
    def _handle_greeting(self, text: str) -> Optional[str]:
        t = (text or "").strip()
        if self._RE_SALAM.match(t):
            return "وعليكم السلام ورحمة الله وبركاته. كيف أقدر أخدمك؟"
        if self._RE_AHLA.match(t):
            return "أهلًا وسهلًا! كيف أقدر أخدمك؟"
        if self._RE_HOW_ARE_YOU.match(t):
            return "الحمد لله بخير. كيف أقدر أخدمك؟"
        if self._RE_THANKS.match(t):
            return "العفو! كيف أقدر أساعدك؟"
        return None

    # -----------------------------------------------------
    # ✅ HTML -> TEXT (remove tags for answers context)
    # -----------------------------------------------------
    def _html_to_text(self, html: str) -> str:
        """
        ✅ Turn HTML chunk into readable plain text (no tags).
        ✅ Preserves newlines (critical to cut single "المادة").
        """
        t = (html or "").strip()
        if not t:
            return ""

        # Remove head/script/style
        t = self._RE_HEAD.sub("\n", t)
        t = self._RE_SCRIPT.sub("\n", t)
        t = self._RE_STYLE.sub("\n", t)

        # Add line breaks / separators
        t = re.sub(r"</h[1-6]>", "\n", t, flags=re.IGNORECASE)
        t = re.sub(r"</p>", "\n", t, flags=re.IGNORECASE)
        t = re.sub(r"</tr>", "\n", t, flags=re.IGNORECASE)
        t = re.sub(r"</td>", " | ", t, flags=re.IGNORECASE)
        t = re.sub(r"</th>", " | ", t, flags=re.IGNORECASE)

        # Remove remaining tags
        t = self._RE_TAGS.sub(" ", t)

        # Decode common entities
        for k, v in self._RE_HTML_ENT.items():
            t = t.replace(k, v)

        t = t.replace("\u00a0", " ")
        t = t.replace("\r\n", "\n").replace("\r", "\n")

        # ✅ Normalize spaces but keep newlines
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"\s*\|\s*", " | ", t)
        t = re.sub(r"\n[ \t]+", "\n", t)
        t = re.sub(r"[ \t]+\n", "\n", t)
        t = re.sub(r"\n{3,}", "\n\n", t).strip()

        return t

    # -----------------------------------------------------
    # ✅ Article phrase extraction + cut single article
    # -----------------------------------------------------
    def _extract_article_phrase(self, text: str) -> str:
        t = (text or "").strip()
        if not t:
            return ""
        m = self._RE_ARTICLE_PHRASE.search(t)
        if not m:
            return ""
        phrase = "المادة " + (m.group(1) or "").strip()
        phrase = re.sub(r"\s+", " ", phrase).strip()
        return phrase[:80]

    def _article_tokens(self, article_phrase: str) -> List[str]:
        ap = self.normalize_arabic(article_phrase or "")
        ap = ap.replace("الماده ", "").replace("المادة ", "").strip()
        toks = re.findall(r"[0-9]+|[\u0600-\u06FF]+", ap)
        toks = [x.strip() for x in toks if x.strip()]
        stop = {"من", "في", "على", "الى", "إلى", "عن", "مع", "و", "او", "أو"}
        toks = [x for x in toks if x not in stop]
        return toks[:8]

    def _cut_single_article(self, full_text: str, article_phrase: str) -> str:
        """
        ✅ If text contains multiple articles, cut from requested article heading to next article heading.
        Works on plain text produced by _html_to_text.
        """
        if not full_text or not article_phrase:
            return ""

        toks = self._article_tokens(article_phrase)
        lines = [ln.strip() for ln in (full_text or "").split("\n") if ln.strip()]
        if not lines:
            return ""

        start = -1
        for i, ln in enumerate(lines):
            n = self.normalize_arabic(ln)
            if "المادة" in n:
                ok = True
                for t in toks:
                    if t not in n:
                        ok = False
                        break
                if ok:
                    start = i
                    break

        if start == -1:
            return ""

        end = len(lines)
        for j in range(start + 1, len(lines)):
            n = self.normalize_arabic(lines[j])
            if "المادة" in n:
                end = j
                break

        out = "\n".join(lines[start:end]).strip()
        return out

    # -----------------------------------------------------
    # ✅ Exec rules extraction (القواعد التنفيذية للمادة ...)
    # -----------------------------------------------------
    def _is_exec_rules_request(self, text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return False
        return bool(self._RE_EXEC_RULES.search(t))

    def _cut_exec_rules_block(self, full_text: str, article_phrase: str) -> str:
        """
        ✅ يقص "القواعد التنفيذية للمادة X" فقط:
        من سطر "القواعد التنفيذية للمادة ..." إلى قبل "المادة ..." التالية.
        """
        if not full_text or not article_phrase:
            return ""

        toks = self._article_tokens(article_phrase)
        lines = [ln.strip() for ln in (full_text or "").split("\n") if ln.strip()]
        if not lines:
            return ""

        # ابحث عن سطر "القواعد التنفيذية للمادة X"
        start = -1
        for i, ln in enumerate(lines):
            n = self.normalize_arabic(ln)
            if "القواعد التنفيذية" in n and "المادة" in n:
                ok = True
                for t in toks:
                    if t not in n:
                        ok = False
                        break
                if ok:
                    start = i
                    break

        if start == -1:
            return ""

        # نهاية البلوك: أول سطر يبدأ بـ "المادة ..." بعده
        end = len(lines)
        for j in range(start + 1, len(lines)):
            n = self.normalize_arabic(lines[j])
            if n.startswith("المادة "):
                end = j
                break

        return "\n".join(lines[start:end]).strip()

    def _extract_exec_rules_only(self, rq: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        ✅ يرجّع القواعد التنفيذية للمادة المطلوبة فقط.
        """
        article_phrase = self._extract_article_phrase(rq)
        if not article_phrase:
            return ""

        docs = self._best_docs(context_docs or [], max_docs=80)
        for d in docs:
            raw = self._safe_str(d.get("content") or "").strip()
            if not raw:
                continue

            txt = self._html_to_text(raw).strip()
            if not txt:
                continue

            cut = self._cut_exec_rules_block(txt, article_phrase)
            if cut:
                if len(cut) > self.article_direct_max_chars:
                    cut = cut[: self.article_direct_max_chars].rstrip() + "…"
                return cut.strip()

        return ""

    # -----------------------------------------------------
    # ✅ HTML post-process: strip fences + sanitize + force valid doc
    # -----------------------------------------------------
    def _strip_code_fences(self, text: str) -> str:
        t = (text or "").strip()
        m = self._RE_CODEFENCE.search(t)
        if m:
            return (m.group(1) or "").strip()
        return t

    def _extract_body_inner(self, html_doc: str) -> str:
        html_doc = (html_doc or "").strip()
        m = re.search(r"<body\b[^>]*>(.*?)</body>", html_doc, flags=re.IGNORECASE | re.DOTALL)
        return (m.group(1) if m else html_doc).strip()

    def _ensure_valid_html_document(self, html_or_fragment: str, title: str) -> str:
        """
        ✅ يضمن Valid HTML document حتى لو دخلنا fragment فقط.
        """
        title = (title or "Document").strip() or "Document"
        frag = (html_or_fragment or "").strip()
        body_inner = self._extract_body_inner(frag)

        out = f"""<!doctype html>
<html lang="ar">
<head>
  <meta charset="utf-8">
  <title>{self._safe_str(title)}</title>
</head>
<body>
{body_inner}
</body>
</html>
""".strip()
        return out

    def _sanitize_html_allowed_tags(self, html_doc: str) -> str:
        """
        ✅ يسمح فقط بتاقات:
        html, head, meta, title, body, h2, h3, p, table, thead, tbody, tr, th, td
        """
        html_doc = (html_doc or "").strip()
        if not html_doc:
            return ""

        # remove script/style entirely
        html_doc = re.sub(r"<script\b.*?</script>", " ", html_doc, flags=re.IGNORECASE | re.DOTALL)
        html_doc = re.sub(r"<style\b.*?</style>", " ", html_doc, flags=re.IGNORECASE | re.DOTALL)
        html_doc = re.sub(r"<!--.*?-->", " ", html_doc, flags=re.DOTALL)

        allowed = {
            "html", "head", "meta", "title", "body",
            "h2", "h3", "p",
            "table", "thead", "tbody", "tr", "th", "td"
        }

        def _tag_repl(m):
            tag = (m.group(1) or "").lower()
            if tag in allowed:
                return m.group(0)
            return " "  # drop tag

        html_doc = re.sub(r"</?\s*([a-zA-Z0-9]+)\b[^>]*>", _tag_repl, html_doc)
        html_doc = re.sub(r"[ \t]+", " ", html_doc)
        html_doc = re.sub(r"\n{3,}", "\n\n", html_doc).strip()
        return html_doc

    # -----------------------------------------------------
    # ✅ Refiner Prompt
    # -----------------------------------------------------
    _REFINER_SYSTEM = """
أنت نظام يفهم اللهجة العامية السعودية/الخليجية ويحوّل سؤال المستخدم إلى صياغة واضحة قابلة للبحث في قاعدة المعرفة الجامعية.

أرجع JSON فقط بدون أي شرح أو نص إضافي.

المفاتيح المطلوبة (بالضبط):
refined_question, intent, entities, constraints, search_queries, needs_clarification, clarifying_question, request_type, file_query

التعليمات:
- refined_question: صياغة عربية فصحى بسيطة وواضحة تعكس قصد المستخدم الحقيقي.
- intent: اختر واحدة من القيم التالية فقط:
  academic_procedure, admission, schedules, tuition, tech_support, general_info, greetings, complaint, other
- entities: قائمة كلمات/كيانات مهمة.
- constraints: قاموس قيود مثل campus/college/date/system_name/level... حسب الحاجة.
- search_queries: 3 إلى 6 عبارات بحث قصيرة ومتنوعة بالعربية تساعد الاسترجاع (RAG).
- needs_clarification: true إذا السؤال ناقص يمنع إجابة صحيحة.
- clarifying_question: سؤال واحد فقط إذا needs_clarification=true، وإلا اتركه فارغاً.
- request_type: "answer" أو "file"
  اجعلها "file" إذا المستخدم يطلب نموذج/ملف/تحميل/رابط/إرسال PDF/DOCX.
- file_query: 2-8 كلمات تمثل اسم الملف المطلوب.
""".strip()

    # -----------------------------------------------------
    # Gemini helpers
    # -----------------------------------------------------
    def _gemini_generate(self, model_name: str, prompt: str, temperature: float, max_tokens: int) -> str:
        if not self.gemini_ready:
            return ""

        def _try(model_to_use: str) -> str:
            model = genai.GenerativeModel(model_to_use)
            gen_cfg = {"temperature": temperature, "max_output_tokens": max_tokens}
            resp = model.generate_content(prompt, generation_config=gen_cfg)
            return (getattr(resp, "text", "") or "").strip()

        try:
            return _try(model_name)
        except Exception as e:
            err = self._safe_str(e)
            print("⚠️ Gemini model failed:", model_name, err)
            fallback_model = "gemini-1.0-pro"
            if model_name.strip().lower() != fallback_model:
                try:
                    out = _try(fallback_model)
                    if out:
                        print("✅ Gemini fallback used:", fallback_model)
                        return out
                except Exception as e2:
                    print("❌ Gemini fallback failed:", self._safe_str(e2))
            return ""

    def _gemini_refine_json(self, user_text: str, context_hint: Optional[str] = None) -> str:
        if not self.gemini_ready:
            return ""
        hint = f"\nسياق إضافي: {context_hint}\n" if context_hint else ""
        prompt = f"""{self._REFINER_SYSTEM}

نص المستخدم (عامي): {user_text}{hint}
أعد JSON فقط.
"""
        text = self._gemini_generate(
            model_name=self.refine_model,
            prompt=prompt,
            temperature=self.refine_temperature,
            max_tokens=self.refine_max_tokens,
        )
        if not text:
            return ""

        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if m:
            return m.group(1).strip()

        m2 = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        if m2:
            return m2.group(1).strip()

        return text

    # -----------------------------------------------------
    # Local fallback refine
    # -----------------------------------------------------
    def _local_fallback_refine(self, user_text: str) -> Dict[str, Any]:
        t = self.normalize_arabic(user_text or "")

        file_kw = [
            "نموذج", "ملف", "تحميل", "تنزيل", "نزّل", "نزل", "رابط",
            "ارسل", "أرسل", "ابي رابط", "pdf", "doc", "docx", "word", "اكسل", "excel"
        ]
        lt = (t or "").lower()
        is_file = any(k in lt for k in file_kw)

        fq = t or ""
        for k in file_kw:
            fq = fq.replace(k, " ")
        fq = re.sub(r"\s+", " ", fq).strip()
        fq = " ".join(fq.split()[:8]).strip()

        return {
            "refined_question": t or (user_text or ""),
            "intent": "other",
            "entities": [],
            "constraints": {},
            "search_queries": [t] if t else ([user_text] if user_text else []),
            "needs_clarification": False,
            "clarifying_question": "",
            "request_type": "file" if is_file else "answer",
            "file_query": fq if is_file else "",
        }

    # -----------------------------------------------------
    # Public refine_query
    # -----------------------------------------------------
    def refine_query(self, user_text: str, context_hint: Optional[str] = None) -> Dict[str, Any]:
        user_text = self.normalize_arabic(user_text or "")

        greet = self._handle_greeting(user_text)
        if greet:
            return {
                "refined_question": user_text,
                "intent": "greetings",
                "entities": [],
                "constraints": {},
                "search_queries": [user_text],
                "needs_clarification": False,
                "clarifying_question": "",
                "request_type": "answer",
                "file_query": "",
                "direct_response": greet,
            }

        content = ""
        if self.refiner_provider == "gemini":
            content = self._gemini_refine_json(user_text, context_hint=context_hint)

        if not content:
            return self._local_fallback_refine(user_text)

        fallback = self._local_fallback_refine(user_text)
        try:
            data = json.loads(content)

            for k in [
                "refined_question",
                "intent",
                "entities",
                "constraints",
                "search_queries",
                "needs_clarification",
                "clarifying_question",
                "request_type",
                "file_query",
            ]:
                if k not in data:
                    data[k] = fallback[k]

            if not isinstance(data.get("entities"), list):
                data["entities"] = []
            if not isinstance(data.get("constraints"), dict):
                data["constraints"] = {}
            if not isinstance(data.get("search_queries"), list):
                data["search_queries"] = []
            if not isinstance(data.get("needs_clarification"), bool):
                data["needs_clarification"] = False
            if not isinstance(data.get("clarifying_question"), str):
                data["clarifying_question"] = ""

            rq = data.get("refined_question") or user_text
            if not isinstance(rq, str) or not rq.strip():
                data["refined_question"] = user_text

            rt = data.get("request_type")
            if rt not in ("answer", "file"):
                data["request_type"] = fallback["request_type"]

            if not isinstance(data.get("file_query"), str):
                data["file_query"] = ""

            return data
        except Exception:
            return fallback

    def build_retrieval_query(self, refined: Dict[str, Any]) -> str:
        qs = refined.get("search_queries") or []
        if isinstance(qs, list) and qs:
            for x in qs:
                sx = self._safe_str(x).strip()
                if sx:
                    return sx
        return (refined.get("refined_question", "") or "").strip()

    # -----------------------------------------------------
    # ✅ HTML conversion (VALID HTML + table aware)
    # -----------------------------------------------------
    def to_structured_html(self, raw_text: str, file_title: Optional[str] = None) -> str:
        raw_text = (raw_text or "").strip()
        if not raw_text:
            return ""

        title = (file_title or "Document").strip() or "Document"

        # Hint: detect markdown table so LLM is pushed to create <table>
        has_md_table = bool(self._RE_MD_TABLE.search(raw_text))

        rules_extra = ""
        if has_md_table:
            rules_extra = "\n- إذا وجدت جدول بصيغة Markdown (علامة | وخط ---) لازم تحوله إلى <table> (thead/tbody/tr/th/td)."

        prompt = f"""
حوّل المحتوى التالي إلى **وثيقة HTML كاملة وصحيحة (Valid HTML Document)**.

FILE_NAME_TITLE (مهم جدًا): {title}

قواعد إلزامية (لا تخالفها):
1) أعد **HTML فقط** بدون أي شرح أو نص خارج HTML أو Markdown.
2) لازم تبدأ بـ: <!doctype html>
3) لازم تحتوي على الترتيب التالي:
   <html lang="ar"> ثم <head> ثم <meta charset="utf-8"> ثم <title> ثم </head> ثم <body> ثم المحتوى ثم </body> ثم </html>
4) <title> لازم يكون مطابق لـ FILE_NAME_TITLE حرفيًا.
5) داخل <body> استخدم فقط هذه التاقات:
   <h2>, <h3>, <p>, <table>, <thead>, <tbody>, <tr>, <th>, <td>
6) ممنوع أي تاقات أخرى (مثل div/span/a/img/script/style).
7) لا تنشئ <table> إلا إذا كان هناك جدول واضح (صفوف/أعمدة) أو كان النص واضح أنه جدول.
{rules_extra}
8) تأكد أن كل التاقات مغلقة بشكل صحيح.

المحتوى:
<<<
{raw_text}
>>>
""".strip()

        # Prefer OpenAI if available unless forced
        use_openai = (self.html_provider == "openai") or (not self.html_provider and self.openai_key and self.llm is not None)

        out = ""

        if use_openai and self.openai_key:
            try:
                llm_html = self._init_openai_chat(
                    model=self.html_model_openai,
                    api_key=self.openai_key,
                    temperature=self.html_temperature,
                )
                if llm_html:
                    msgs = [
                        SystemMessage(content="أعد HTML فقط. ممنوع أي نص خارج HTML. ممنوع Markdown."),
                        HumanMessage(content=prompt),
                    ]
                    resp = llm_html.invoke(msgs)
                    out = self._safe_str(getattr(resp, "content", "")).strip()
            except Exception as e:
                print("⚠️ OpenAI HTML convert failed:", self._safe_str(e))

        if not out and self.gemini_ready:
            try:
                out = self._gemini_generate(
                    model_name=self.html_model_gemini,
                    prompt=prompt,
                    temperature=self.html_temperature,
                    max_tokens=self.html_max_tokens,
                )
            except Exception as e:
                print("⚠️ Gemini HTML convert failed:", self._safe_str(e))
                out = ""

        # ✅ HARD guarantee (even if model returns fragment/garbage)
        out = self._strip_code_fences(out)
        out = self._ensure_valid_html_document(out, title)
        out = self._sanitize_html_allowed_tags(out)
        out = self._ensure_valid_html_document(out, title)

        return (out or "").strip()

    # -----------------------------------------------------
    # Context + sources
    # -----------------------------------------------------
    def _best_docs(self, context_docs: List[Dict], max_docs: int) -> List[Dict]:
        docs = context_docs or []

        def score_key(d):
            sc = d.get("score")
            if isinstance(sc, (int, float)):
                return (0, float(sc))
            return (1, 1e9)

        docs_sorted = sorted([d for d in docs if isinstance(d, dict)], key=score_key)
        return docs_sorted[:max_docs]

    def _compress_text(self, text: str, max_chars: int) -> str:
        t = (text or "").strip()
        t = re.sub(r"\s+", " ", t).strip()
        if len(t) <= max_chars:
            return t
        return t[:max_chars].rstrip() + "…"

    def _looks_like_markdown_table(self, text: str) -> bool:
        t = (text or "")
        if "|" not in t:
            return False
        if self._RE_MD_TABLE.search(t):
            return True
        if ("---" in t) and ("|" in t):
            return True
        return False

    def _looks_like_html_table(self, html: str) -> bool:
        t = (html or "").lower()
        return "<table" in t and "</table>" in t

    def _pick_best_doc(self, context_docs: List[Dict]) -> Optional[Dict]:
        docs = [d for d in (context_docs or []) if isinstance(d, dict)]
        if not docs:
            return None

        def score_key(d):
            sc = d.get("score")
            if isinstance(sc, (int, float)):
                return (0, float(sc))
            return (1, 1e9)

        docs.sort(key=score_key)
        return docs[0] if docs else None

    def _should_return_table_directly(self, context_docs: List[Dict]) -> Optional[str]:
        """
        ✅ إذا أفضل نتيجة جدول: رجّعها "كنص جدول" بدون HTML tags
        عشان ما يطلع للمستخدم تاقات.
        """
        best = self._pick_best_doc(context_docs or [])
        if not best:
            return None

        meta = best.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}

        is_table_meta = bool(meta.get("is_table")) or bool(best.get("is_full_table"))
        content = self._safe_str(best.get("content") or "").strip()
        if not content:
            return None

        # detect by meta OR by html
        if is_table_meta or self._looks_like_html_table(content) or self._looks_like_markdown_table(content):
            txt = self._html_to_text(content)
            return txt.strip() if txt.strip() else content.strip()

        return None

    def _build_context(self, context_docs: List[Dict]) -> str:
        picked = self._best_docs(context_docs or [], self.max_docs_in_context)

        blocks: List[str] = []
        used = 0

        for i, doc in enumerate(picked, 1):
            raw_content = self._safe_str(doc.get("content", "")).strip()
            if not raw_content:
                continue

            meta = doc.get("metadata") or {}
            if not isinstance(meta, dict):
                meta = {}

            source = (
                meta.get("original_name")
                or meta.get("filename")
                or meta.get("doc_key")
                or f"مصدر {i}"
            )
            source = self._clean_source_name(self._safe_str(source)) or f"مصدر {i}"

            content_no_tags = self._html_to_text(raw_content)

            if self._looks_like_markdown_table(content_no_tags):
                chunk = content_no_tags.strip()
            else:
                chunk = self._compress_text(content_no_tags, self.max_chars_per_doc)

            block = f"[{i}] {source}\n{chunk}"
            if used + len(block) > self.max_total_context_chars:
                break

            blocks.append(block)
            used += len(block)

        return "\n\n".join(blocks).strip()

    def extract_sources(self, context_docs: List[Dict], max_sources: int = 4) -> List[str]:
        sources: List[str] = []
        seen = set()

        for doc in self._best_docs(context_docs or [], max_docs=max_sources * 6):
            meta = doc.get("metadata") or {}
            if not isinstance(meta, dict):
                meta = {}

            name = (
                meta.get("original_name")
                or meta.get("filename")
                or meta.get("doc_name")
                or meta.get("doc_key")
                or meta.get("source")
                or meta.get("path")
            )

            name = self._clean_source_name(self._safe_str(name))
            if not name:
                continue

            key = name.lower().strip()
            if key in seen:
                continue
            seen.add(key)

            sources.append(name)
            if len(sources) >= max_sources:
                break

        return sources

    def append_sources_to_answer(self, answer: str, context_docs: List[Dict]) -> str:
        ans = (answer or "").strip()
        if not self.append_sources:
            return ans

        if re.search(r"\bالمصادر\s*:", ans):
            return ans

        srcs = self.extract_sources(context_docs or [], max_sources=self.max_sources)
        if not srcs:
            return ans

        lines = ["", "المصادر:"]
        for i, s in enumerate(srcs, 1):
            lines.append(f"- [{i}] {s}")
        return (ans + "\n" + "\n".join(lines)).strip()

    # -----------------------------------------------------
    # ✅ Article direct return (ONLY requested article)
    # -----------------------------------------------------
    def _is_article_request(self, text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return False
        return bool(self._RE_ARTICLE_WORD.search(t))

    def _extract_requested_article_only(self, rq: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        ✅ يرجّع فقط المادة المطلوبة (بدقة) بدل تجميع مواد كثيرة.
        """
        article_phrase = self._extract_article_phrase(rq)
        if not article_phrase:
            return ""

        docs = self._best_docs(context_docs or [], max_docs=60)
        for d in docs:
            raw = self._safe_str(d.get("content") or "").strip()
            if not raw:
                continue

            txt = self._html_to_text(raw).strip()
            if not txt:
                continue

            cut = self._cut_single_article(txt, article_phrase)
            if cut:
                if len(cut) > self.article_direct_max_chars:
                    cut = cut[: self.article_direct_max_chars].rstrip() + "…"
                return cut.strip()

        return ""

    # -----------------------------------------------------
    # Gemini fallback answer
    # -----------------------------------------------------
    def _gemini_answer_from_context(self, refined_question: str, context_text: str, user_role: str, intent: Optional[str] = None) -> str:
        if not self.gemini_ready:
            return "عذراً، نظام الذكاء الاصطناعي غير متصل حالياً."

        intent_line = f"نوع الطلب: {intent}" if intent else "نوع الطلب: غير محدد"

        prompt = f"""
أنت مساعد رسمي لجامعة الأمير سطام.
دور المستخدم: {user_role}
{intent_line}

تعليمات إلزامية:
- استخدم فقط المعلومات الموجودة في "النص الرسمي المعتمد" أدناه.
- إذا لم تجد إجابة في النص، قل: "عذراً، هذه المعلومة غير متوفرة في الملفات المرفوعة حالياً."
- لا تذكر كلمات مثل (سياق، ملف، مستند، RAG).
- اكتب بالعربية الفصحى وبأسلوب رسمي.
- ممنوع إرجاع أي تاقات HTML أو أكواد.

النص الرسمي المعتمد:
{context_text}

سؤال المستخدم:
{refined_question}
""".strip()

        out = self._gemini_generate(
            model_name=self.fallback_answer_model,
            prompt=prompt,
            temperature=0.2,
            max_tokens=1024,
        )
        return out.strip() or "عذراً، لم أتمكن من استخراج إجابة من الملفات."

    # -----------------------------------------------------
    # Answer generation
    # -----------------------------------------------------
    def generate_response(
        self,
        user_query: str,
        context_docs: Optional[List[Dict]] = None,
        user_role: str = "طالب",
        system_message: Optional[str] = None,
        refined_question: Optional[str] = None,
        intent: Optional[str] = None,
    ) -> str:
        user_query = (user_query or "").strip()
        if not user_query:
            return "اكتب سؤالك من فضلك."

        greet = self._handle_greeting(user_query)
        if greet:
            return greet

        rq = (refined_question or user_query).strip()

        # ✅ إذا يطلب "القواعد التنفيذية للمادة ..." رجّعها مباشرة (بدون LLM)
        if self._is_exec_rules_request(rq) and self._is_article_request(rq):
            direct_rules = self._extract_exec_rules_only(rq, context_docs or [])
            if direct_rules:
                return self.append_sources_to_answer(direct_rules, context_docs or [])

        # ✅ لو سؤال "المادة ..." حاول ترجع المادة فقط،
        # وإذا فشل لا توقف — كمل للمسار العادي عشان LLM يجاوب من السياق
        if self._is_article_request(rq):
            direct = self._extract_requested_article_only(rq, context_docs or [])
            if direct:
                return self.append_sources_to_answer(direct, context_docs or [])
            # لا ترجع "غير متوفرة" هنا

        # ✅ direct table return (يميز الجدول)
        direct_table = self._should_return_table_directly(context_docs or [])
        if direct_table:
            return self.append_sources_to_answer(direct_table, context_docs or [])

        context_text = self._build_context(context_docs or [])
        if not context_text:
            return "عذراً، هذه المعلومة غير متوفرة في الملفات المرفوعة حالياً."

        base_system = (system_message or "").strip()
        role_line = f"دور المستخدم: {user_role}".strip()
        intent_line = f"نوع الطلب: {intent}".strip() if intent else "نوع الطلب: غير محدد"

        rules = f"""
أنت مساعد رسمي لجامعة الأمير سطام.

{role_line}
{intent_line}

تعليمات إلزامية:
1) استخدم فقط المعلومات الموجودة في "النص الرسمي المعتمد" أدناه.
2) مسموح بالاستنتاج المنطقي المباشر إذا كان مبنيًا بوضوح على نفس الجملة/الفقرة في النص.
3) ممنوع إدخال أي معلومات من خارج النص.
4) إذا لم تجد في النص ما يجيب، اكتب حرفيًا:
   "عذراً، هذه المعلومة غير متوفرة في الملفات المرفوعة حالياً."
5) لا تذكر كلمات مثل (سياق، ملف، مستند، RAG).
6) اكتب بالعربية الفصحى وبأسلوب رسمي.
7) ممنوع إرجاع أي تاقات HTML أو أكواد.

النص الرسمي المعتمد:
{context_text}
""".strip()

        final_system = (base_system + "\n\n" + rules).strip() if base_system else rules
        messages = [SystemMessage(content=final_system), HumanMessage(content=rq)]

        if not self.llm:
            ans = self._gemini_answer_from_context(rq, context_text, user_role, intent=intent)
            return self.append_sources_to_answer(ans, context_docs or [])

        try:
            llm = self.llm
            try:
                llm = llm.bind(temperature=self.answer_temperature)
            except Exception:
                pass

            response = llm.invoke(messages)
            text = self._safe_str(getattr(response, "content", "")).strip()
            text = text or "عذراً، لم أتمكن من استخراج إجابة من الملفات."
            return self.append_sources_to_answer(text, context_docs or [])

        except Exception as e:
            err = self._safe_str(e)
            print("LLM invoke error:", err)
            traceback.print_exc()

            if self.retry_on_429 and ("429" in err or "rate limit" in err.lower()):
                time.sleep(self.retry_sleep_seconds)
                try:
                    response = self.llm.invoke(messages)
                    text = self._safe_str(getattr(response, "content", "")).strip()
                    text = text or "عذراً، لم أتمكن من استخراج إجابة من الملفات."
                    return self.append_sources_to_answer(text, context_docs or [])
                except Exception:
                    pass

            ans = self._gemini_answer_from_context(rq, context_text, user_role, intent=intent)
            return self.append_sources_to_answer(ans, context_docs or [])

    def is_available(self) -> bool:
        return (self.llm is not None) or self.gemini_ready