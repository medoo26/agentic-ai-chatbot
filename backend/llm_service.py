# llm_service.py
import os
import re
import sys
import json
import time
import traceback
import warnings
from typing import List, Dict, Optional, Any, Iterator

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Gemini SDK (import errors are surfaced in _init_gemini — bare except hides broken protobuf)
GEMINI_IMPORT_ERROR: Optional[BaseException] = None
warnings.filterwarnings(
    "ignore",
    message=r"All support for the `google\.generativeai` package has ended\..*",
)
try:
    import google.generativeai as genai
except Exception as _e:
    genai = None
    GEMINI_IMPORT_ERROR = _e


def _gemini_import_explain() -> str:
    if GEMINI_IMPORT_ERROR is None:
        return "google.generativeai did not import (unknown)."
    e = GEMINI_IMPORT_ERROR
    msg = str(e).strip()
    tail = ""
    low = msg.lower()
    if "any_pb2" in msg or "protobuf" in low:
        tail = (
            " On Windows this usually means `protobuf` in the active venv is corrupted or locked "
            "(often after a failed pip uninstall). Stop uvicorn and any Python using that venv, then run: "
            "`python -m pip install --force-reinstall \"protobuf>=5.29,<7\" google-generativeai`."
        )
    elif isinstance(e, ModuleNotFoundError):
        tail = " Run `python -m pip install google-generativeai` in the same environment that starts uvicorn."
    return f"{type(e).__name__}: {msg}{tail}"

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
    _RE_CANON_TABLE = re.compile(r"^\s*Table:\s*(.+)$", re.IGNORECASE)

    # ✅ يلتقط المادة بكل أشكالها: "المادة 4", "للمادة 4", "للمادة الخامسة", "بمادة 5"
    _RE_ARTICLE_PHRASE = re.compile(
        r"(?:لل|ل|بال|وال|فال)?م[اآ]د[هة]\s+([^\n\.<،,:؛!\?؟]{1,80})",
        re.IGNORECASE,
    )
    _RE_NO_INFO_ANSWER = re.compile(
        r"عذراً[،,]?\s*.*?غير\s+متوفرة(?:\s+في\s+الملفات\s+المرفوعة)?\s+حالياً",
        re.IGNORECASE,
    )

    # ✅ wants_exec_rules detection (query-side)
    _RE_EXEC_RULES_Q = re.compile(r"\bالقواعد\s+التنفيذية\b", re.IGNORECASE)
    _RE_FILE_INTENT = re.compile(
        r"(?:\b(?:اريد|أريد|ابغى|أبغى|احتاج|أحتاج|اعطني|أعطني|ارسل|أرسل|نزّل|نزل|هات)\b.*\b(?:ملف|مستند|وثيقة|لائحة|نموذج|pdf|docx|doc)\b)"
        r"|(?:\b(?:ملف|مستند|وثيقة|لائحة|نموذج)\b.*\b(?:اريد|أريد|ابغى|أبغى|احتاج|أحتاج|اعطني|أعطني|ارسل|أرسل|نزّل|نزل|هات)\b)",
        re.IGNORECASE,
    )

    SYSTEM_PROMPT = """You are a specialized Document Assistant. Your sole task is to extract text and convert it into clean, structured HTML. Keep the logic of paging. Extract text from the attached file as it is.

STRICT RULES:

1. No Commentary: Do not include any introductory or concluding text.
2. No Metadata: Ignore watermarks, stamps, headers, footers.
3. Preserve Structure: Use <table> tags for tables, <h1>-<h3> for titles, and <p> for paragraphs.
4. Clean HTML: Use only basic tags (<html>, <body>, <h1>-<h3>, <p>, <table>, <thead>, <tbody>, <tr>, <th>, <td>). Do not include <style>, <script>, <nav>, or <footer>. No CSS or inline styles. Return a valid HTML page.
5. Accuracy: Extract text exactly as written. Do not correct grammar or summarize.
6. Include <!DOCTYPE html> and use lang="ar" dir="rtl" when content is Arabic.
7. Extract table contents exactly as they appear, without adding inferred structure.
8. Include page numbers in page tag (attribute-based), preserving page logic.
9. Document title to use in <title>: {title}.
"""

    SYSTEM_PROMPT_BATCHING = """You are a specialized Document Assistant. Your sole task is to extract text and convert it into clean, structured HTML. Keep the logic of paging. Extract text from the attached file as it is.

STRICT RULES FOR BATCHING:

1. No Commentary: Do not include any introductory or concluding text.
2. No Metadata: Ignore watermarks, stamps, headers, footers.
3. Preserve Structure: Use <table> tags for tables, <h1>-<h3> for titles, and <p> for paragraphs.
4. Return ONLY Content: Return only content inside <div> tags. Do NOT include <html>, <body>, <head>, <!DOCTYPE>, or wrapper tags.
5. Page Tags: Use <page number="X"> tags to mark each page.
6. Clean HTML: Use only basic tags: <div>, <h1>-<h3>, <p>, <table>, <tr>, <td>, <th>, etc. No CSS or inline styles.
7. Accuracy: Extract text exactly as written. Do not correct grammar or summarize.
8. Extract table contents as-is; do not add or suggest structure.
9. Include page numbers in page tag (attribute-based).
10. End Detection: If document has fewer pages than requested, set reached_end=true and last_page to the highest extracted page.
"""

    def __init__(self):
        # Providers
        self.refiner_provider = (os.getenv("REFINER_PROVIDER") or "gemini").strip().lower()
        self.llm_provider = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()

        # Models
        self.chat_model = (os.getenv("CHAT_MODEL") or "gpt-4o-mini").strip()
        self.refine_model = (os.getenv("REFINE_MODEL") or "models/gemini-flash-latest").strip()
        self.fallback_answer_model = (os.getenv("FALLBACK_ANSWER_MODEL") or "models/gemini-flash-latest").strip()

        # Keys (Gemini accepts either name; GOOGLE_API_KEY matches Google AI Studio)
        self.openai_key = self._extract_openai_key((os.getenv("OPENAI_API_KEY") or "").strip())
        self.gemini_key = (
            (os.getenv("GEMINI_API_KEY") or "").strip()
            or (os.getenv("GOOGLE_API_KEY") or "").strip()
        )

        # Configs
        self.refine_max_tokens = int(os.getenv("REFINE_MAX_TOKENS") or "256")
        self.refine_temperature = float(os.getenv("REFINE_TEMPERATURE") or "0.0")
        self.refine_retry_attempts = int(os.getenv("REFINE_RETRY_ATTEMPTS") or "2")
        self.answer_temperature = float(os.getenv("ANSWER_TEMPERATURE") or "0.2")

        # Context sizing
        self.max_docs_in_context = int(os.getenv("MAX_DOCS_IN_CONTEXT") or "24")
        self.max_chars_per_doc = int(os.getenv("MAX_CHARS_PER_DOC") or "4000")
        self.max_total_context_chars = int(os.getenv("MAX_TOTAL_CONTEXT_CHARS") or "10000")

        # Article direct output sizing (for "نص المادة")
        self.article_direct_max_chars = int(os.getenv("ARTICLE_DIRECT_MAX_CHARS") or "20000")

        # Sources
        self.append_sources = (os.getenv("APPEND_SOURCES", "0").strip() == "1")
        self.max_sources = int(os.getenv("MAX_SOURCES") or "4")
        self.last_answer_source_titles: List[str] = []

        # Rate-limit backoff
        self.retry_on_429 = int(os.getenv("RETRY_ON_429") or "1")
        self.retry_sleep_seconds = float(os.getenv("RETRY_SLEEP_SECONDS") or "2.0")

        # HTML conversion configs
        self.html_provider = (os.getenv("HTML_PROVIDER") or "").strip().lower()
        self.html_model_openai = (os.getenv("HTML_MODEL_OPENAI") or self.chat_model).strip()
        self.html_model_gemini = (
            (os.getenv("GEMINI_MODEL_NAME") or "").strip()
            or (os.getenv("HTML_MODEL_GEMINI") or "").strip()
            or self.fallback_answer_model
        ).strip()
        # Whole-document HTML can be huge; 2048 tokens truncates long PDFs mid-tag (Gemini PDF path).
        self.html_max_tokens = int(os.getenv("HTML_MAX_TOKENS") or "8192")
        # One-shot PDF→HTML via Files API needs a high ceiling (model may cap lower).
        self.html_max_tokens_pdf = int(os.getenv("HTML_MAX_TOKENS_PDF") or "65536")
        self.html_temperature = float(os.getenv("HTML_TEMPERATURE") or "0.0")
        # Gemini PDF path only: extra generate_content rounds when finish_reason is MAX_TOKENS (full HTML for ingestion; no OpenAI).
        self.gemini_pdf_max_continuation_rounds = int(
            os.getenv("GEMINI_PDF_MAX_CONTINUATION_ROUNDS") or "16"
        )
        self.gemini_pdf_continuation_tail_chars = int(
            os.getenv("GEMINI_PDF_CONTINUATION_TAIL_CHARS") or "12000"
        )
        # Gemini 2.5 models count hidden "thinking" tokens toward max_output_tokens,
        # which can truncate short classification JSON. Disable thinking for deterministic calls.
        self.gemini_disable_thinking = (os.getenv("GEMINI_DISABLE_THINKING") or "1").strip() == "1"

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
    def _init_openai_chat(
        self,
        model: str,
        api_key: str,
        temperature: float,
        max_tokens: Optional[int] = None,
    ):
        extra: Dict[str, Any] = {}
        if max_tokens is not None:
            extra["max_tokens"] = max_tokens
        configs = [
            dict(model=model, openai_api_key=api_key, temperature=temperature, **extra),
            dict(model_name=model, openai_api_key=api_key, temperature=temperature, **extra),
            dict(model=model, api_key=api_key, temperature=temperature, **extra),
            dict(model_name=model, api_key=api_key, temperature=temperature, **extra),
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
            print(f"⚠️ Google Generative AI SDK unavailable — {_gemini_import_explain()}")
            self.gemini_ready = False
            return
        if not self.gemini_key:
            print("⚠️ GEMINI_API_KEY or GOOGLE_API_KEY missing (check .env).")
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
    # ✅ HTML -> TEXT
    # -----------------------------------------------------
    def _html_to_text(self, html: str) -> str:
        t = (html or "").strip()
        if not t:
            return ""
        t = self._RE_HEAD.sub("\n", t)
        t = self._RE_SCRIPT.sub("\n", t)
        t = self._RE_STYLE.sub("\n", t)
        t = re.sub(r"</h[1-6]>", "\n", t, flags=re.IGNORECASE)
        t = re.sub(r"</p>", "\n", t, flags=re.IGNORECASE)
        t = re.sub(r"</tr>", "\n", t, flags=re.IGNORECASE)
        t = re.sub(r"</td>", " | ", t, flags=re.IGNORECASE)
        t = re.sub(r"</th>", " | ", t, flags=re.IGNORECASE)
        t = self._RE_TAGS.sub(" ", t)
        for k, v in self._RE_HTML_ENT.items():
            t = t.replace(k, v)
        t = t.replace("\u00a0", " ")
        t = t.replace("\r\n", "\n").replace("\r", "\n")
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
        phrase = re.split(r"\s+(?:من|في|عن|الفصل)\b", phrase)[0].strip()
        phrase = re.sub(r"\s+", " ", phrase).strip()
        return phrase[:80]

    def _looks_like_file_request(self, text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return False
        return bool(self._RE_FILE_INTENT.search(t))

    def _build_file_query_from_text(self, text: str) -> str:
        t = self.normalize_arabic(text or "")
        t = re.sub(
            r"\b(?:اريد|أريد|ابغى|أبغى|احتاج|أحتاج|اعطني|أعطني|ارسل|أرسل|نزّل|نزل|هات|لو|ممكن|فضلا|فضلاً)\b",
            " ",
            t,
            flags=re.IGNORECASE,
        )
        t = re.sub(
            r"\b(?:ملف|مستند|وثيقة|نموذج|تحميل|رابط|pdf|docx|doc)\b",
            " ",
            t,
            flags=re.IGNORECASE,
        )
        t = re.sub(r"\s+", " ", t).strip()
        toks = t.split()
        if not toks:
            return ""
        return " ".join(toks[:8]).strip()

    def _article_tokens(self, article_phrase: str) -> List[str]:
        ap = self.normalize_arabic(article_phrase or "")
        ap = ap.replace("الماده ", "").replace("المادة ", "").strip()
        toks = re.findall(r"[0-9]+|[\u0600-\u06FF]+", ap)
        toks = [x.strip() for x in toks if x.strip()]
        stop = {"من", "في", "على", "الى", "إلى", "عن", "مع", "و", "او", "أو"}
        toks = [x for x in toks if x not in stop]
        return toks[:8]

    def _cut_single_article(self, full_text: str, article_phrase: str) -> str:
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
    # ✅ Exec rules extraction
    # -----------------------------------------------------
    def _is_exec_rules_request(self, text: str) -> bool:
        return bool(self._RE_EXEC_RULES.search((text or "").strip()))

    def _cut_exec_rules_block(self, full_text: str, article_phrase: str) -> str:
        if not full_text or not article_phrase:
            return ""
        toks = self._article_tokens(article_phrase)
        lines = [ln.strip() for ln in (full_text or "").split("\n") if ln.strip()]
        if not lines:
            return ""
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
        end = len(lines)
        for j in range(start + 1, len(lines)):
            n = self.normalize_arabic(lines[j])
            if n.startswith("المادة "):
                end = j
                break
        return "\n".join(lines[start:end]).strip()

    def _extract_exec_rules_only(self, rq: str, context_docs: List[Dict[str, Any]]) -> str:
        article_phrase = self._extract_article_phrase(rq)
        if not article_phrase:
            return ""
        docs = sorted(
            self._best_docs(context_docs or [], max_docs=80),
            key=self._chunk_sort_key,
        )
        parts: List[str] = []
        for d in docs:
            raw = self._safe_str(d.get("content") or "").strip()
            if not raw:
                continue
            txt = self._html_to_text(raw).strip()
            if txt:
                parts.append(txt)
        merged = "\n\n".join(parts).strip()
        if not merged:
            return ""
        cut = self._cut_exec_rules_block(merged, article_phrase)
        if not cut:
            return ""
        if len(cut) > self.article_direct_max_chars:
            cut = cut[: self.article_direct_max_chars].rstrip() + "…"
        return cut.strip()

    # -----------------------------------------------------
    # ✅ HTML helper (strip code fences only)
    # -----------------------------------------------------
    def _strip_code_fences(self, text: str) -> str:
        t = (text or "").strip()
        m = self._RE_CODEFENCE.search(t)
        if m:
            return (m.group(1) or "").strip()
        return t


    # -----------------------------------------------------
    # ✅ Refiner Prompt
    # -----------------------------------------------------
    _REFINER_SYSTEM = """
أنت مُحلّل نية لروبوت جامعي. مهمتك تحويل سؤال المستخدم (حتى لو كان عاميًا) إلى JSON صالح للمعالجة الخلفية.

أعد JSON فقط. ممنوع أي شرح أو Markdown أو نص قبل/بعد JSON.

المخطط الإلزامي (بنفس الأسماء والترتيب):
1) refined_question: string
2) request_type: "answer" | "file"
3) file_query: string
4) is_followup: boolean

قواعد كل حقل:
- refined_question:
  صياغة عربية فصحى واضحة ومباشرة تعكس طلب المستخدم الحقيقي.

- request_type:
  "file" فقط عندما يطلب المستخدم صراحة ملفًا/مستندًا/نموذجًا.
  غير ذلك: "answer".

- file_query:
  إذا request_type="file": من 2 إلى 8 كلمات تمثل اسم الملف المطلوب.
  احذف الكلمات العامة مثل: ملف، مستند، نموذج، تحميل، رابط، PDF، DOCX.
  إذا request_type="answer": قيمة فارغة "".

- is_followup:
  true إذا كان سؤال المستخدم متابعة مباشرة لآخر تبادل في المحادثة (يعتمد على الضمائر/الإحالة/الاختصار).
  false إذا كان سؤالًا جديدًا مستقلًا.

قواعد دمج التاريخ:
- إذا is_followup=true: اجعل refined_question سؤالًا مستقلًا مفهومًا بإدماج معلومات آخر تبادل.
- إذا is_followup=false: لا تدمج التاريخ، واعتبر السؤال جديدًا.

قيود صارمة:
- لا تضف أي مفاتيح إضافية.
- لا تحذف أي مفتاح من المفاتيح الخمسة.
- يجب أن يكون الناتج JSON صالحًا (double quotes).

مثال إخراج صحيح:
{
  "refined_question": "أحتاج لائحة الدراسة والاختبارات للمرحلة الجامعية",
  "request_type": "file",
  "file_query": "لائحة الدراسة والاختبارات المرحلة الجامعية",
  "is_followup": false
}
""".strip()


    # -----------------------------------------------------
    # Gemini helpers
    # -----------------------------------------------------
    def _gemini_generate(self, model_name: str, prompt: str, temperature: float, max_tokens: int) -> str:
        if not self.gemini_ready:
            return ""

        def _try(model_to_use: str) -> str:
            model = genai.GenerativeModel(model_to_use)
            gen_cfg: Dict[str, Any] = {"temperature": temperature, "max_output_tokens": max_tokens}
            if self.gemini_disable_thinking:
                gen_cfg["thinking_config"] = {"thinking_budget": 0}
            try:
                resp = model.generate_content(prompt, generation_config=gen_cfg)
            except Exception as e:
                if "thinking_config" in gen_cfg and "thinking" in self._safe_str(e).lower():
                    gen_cfg.pop("thinking_config", None)
                    resp = model.generate_content(prompt, generation_config=gen_cfg)
                else:
                    raise
            try:
                cands = getattr(resp, "candidates", None) or []
                if cands:
                    fr = getattr(cands[0], "finish_reason", None)
                    fr_name = getattr(fr, "name", None) or str(fr)
                    if fr_name and "MAX" in str(fr_name).upper() and "TOKEN" in str(fr_name).upper():
                        print(f"[TRACE][GEMINI] finish_reason=MAX_TOKENS | model={model_to_use} | max_output_tokens={max_tokens}")
            except Exception:
                pass
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

    def _gemini_refine_json(
        self,
        user_text: str,
        context_hint: Optional[str] = None,
        previous_turn: Optional[Dict[str, str]] = None,
    ) -> str:
        if not self.gemini_ready:
            return ""
        hint = f"\nسياق إضافي: {context_hint}\n" if context_hint else ""
        prev_user = self._safe_str((previous_turn or {}).get("user") or "").strip()
        prev_assistant = self._safe_str((previous_turn or {}).get("assistant") or "").strip()
        history_block = ""
        if prev_user and prev_assistant:
            history_block = f"""

آخر تبادل في المحادثة:
- المستخدم: {prev_user}
- المساعد: {prev_assistant}
"""
        prompt = f"""{self._REFINER_SYSTEM}

نص المستخدم (عامي): {user_text}{hint}{history_block}
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
    # Public refine_query
    # -----------------------------------------------------
    def refine_query(
        self,
        user_text: str,
        context_hint: Optional[str] = None,
        previous_turn: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        user_text = self.normalize_arabic(user_text or "")

        greet = self._handle_greeting(user_text)
        if greet:
            return {
                "refined_question": user_text,
                "request_type": "answer",
                "file_query": "",
                "is_followup": False,
                "direct_response": greet,
            }

        heuristic_file = self._looks_like_file_request(user_text)
        heuristic_file_query = self._build_file_query_from_text(user_text) if heuristic_file else ""
        fallback: Dict[str, Any] = {
            "refined_question": user_text,
            "request_type": "file" if heuristic_file else "answer",
            "file_query": heuristic_file_query,
            "is_followup": False,
        }
        attempts = max(1, self.refine_retry_attempts)
        for attempt in range(1, attempts + 1):
            content = ""
            if self.refiner_provider == "gemini":
                content = self._gemini_refine_json(
                    user_text, context_hint=context_hint, previous_turn=previous_turn
                )
                print(
                    "[TRACE][REFINE] classification_model_output",
                    {
                        "provider": self.refiner_provider,
                        "model": self.refine_model,
                        "attempt": attempt,
                        "raw_output": content,
                    },
                )
            if not content:
                print(f"[TRACE][REFINE] empty_output | attempt={attempt}/{attempts}")
                continue
            try:
                data = json.loads(content)
                for k in [
                    "refined_question",
                    "request_type", "file_query",
                    "is_followup",
                ]:
                    if k not in data:
                        data[k] = fallback[k]

                rq = data.get("refined_question") or user_text
                if not isinstance(rq, str) or not rq.strip():
                    data["refined_question"] = user_text

                rt = data.get("request_type")
                if rt not in ("answer", "file"):
                    data["request_type"] = fallback["request_type"]

                if not isinstance(data.get("file_query"), str):
                    data["file_query"] = ""
                if not isinstance(data.get("is_followup"), bool):
                    data["is_followup"] = bool(fallback.get("is_followup", False))

                if data.get("request_type") == "file" and not (data.get("file_query") or "").strip():
                    data["file_query"] = heuristic_file_query or self._build_file_query_from_text(data.get("refined_question") or "")

                print(
                    "[TRACE][REFINE] parsed_ok",
                    {
                        "attempt": attempt,
                        "request_type": data.get("request_type"),
                        "file_query": data.get("file_query"),
                        "is_followup": data.get("is_followup"),
                    },
                )
                return data
            except Exception as e:
                print(
                    "[TRACE][REFINE] parse_failed",
                    {"attempt": attempt, "error": self._safe_str(e)},
                )
                continue

        print(
            "[TRACE][REFINE] fallback_used",
            {
                "reason": "invalid_or_empty_refiner_json",
                "request_type": fallback["request_type"],
                "file_query": fallback["file_query"],
            },
        )
        return fallback

    def build_retrieval_query(self, refined: Dict[str, Any]) -> str:
        request_type = self._safe_str(refined.get("request_type") or "").strip().lower()
        file_query = self._safe_str(refined.get("file_query") or "").strip()
        refined_question = self._safe_str(refined.get("refined_question") or "").strip()

        if request_type == "file" and file_query:
            return file_query
        if refined_question:
            return refined_question
        if file_query:
            return file_query
        return ""

    # -----------------------------------------------------
    # ✅ HTML conversion (NO postprocess fixes after LLM)
    # -----------------------------------------------------
    def to_structured_html(self, raw_text: str, file_title: Optional[str] = None) -> str:
        raw_text = (raw_text or "").strip()
        if not raw_text:
            print("[TRACE][HTML_CONVERT] skip_empty_input")
            return ""

        title = (file_title or "Document").strip() or "Document"
        has_md_table = bool(self._RE_MD_TABLE.search(raw_text))

        rules_extra = ""
        if has_md_table:
            rules_extra = (
                "\n- إذا وجدت جدول بصيغة Markdown (علامة | وخط ---) لازم تحوله إلى "
                "<table><thead><tbody><tr><th><td> بشكل صحيح."
            )

        base_prompt = self.SYSTEM_PROMPT.format(title=title).strip()
        prompt = f"""{base_prompt}

Additional constraints:
- Return full valid HTML document only.
- Keep legal hierarchy when present (chapter/section/article) using h1/h2/h3.
- If Markdown table exists, convert it into valid HTML table structure.
{rules_extra}

CONTENT:
<<<
{raw_text}
>>>
""".strip()

        use_openai = (
            (self.html_provider == "openai")
            or (not self.html_provider and self.openai_key and self.llm is not None)
        )
        provider_mode = (self.html_provider or "auto").strip().lower()
        print(
            "[TRACE][HTML_CONVERT] start "
            f"| provider={provider_mode} "
            f"| use_openai={bool(use_openai)} "
            f"| openai_key={bool(self.openai_key)} "
            f"| openai_llm_ready={self.llm is not None} "
            f"| gemini_ready={bool(self.gemini_ready)} "
            f"| input_chars={len(raw_text)}"
        )

        openai_ready = bool(self.openai_key and (self.llm is not None))
        gemini_ready = bool(self.gemini_ready)
        no_provider_ready = (
            (provider_mode == "openai" and not openai_ready)
            or (provider_mode == "gemini" and not gemini_ready)
            or (provider_mode == "auto" and not openai_ready and not gemini_ready)
        )
        if no_provider_ready:
            msg = (
                "HTML conversion provider is unavailable. "
                f"provider={provider_mode}, "
                f"openai_ready={openai_ready}, gemini_ready={gemini_ready}. "
                "Fix provider setup (API key/dependencies) before upload indexing."
            )
            print(f"[TRACE][HTML_CONVERT] unavailable | {msg}")
            raise RuntimeError(msg)

        out = ""

        # OpenAI
        if use_openai and self.openai_key:
            try:
                llm_html = self._init_openai_chat(
                    model=self.html_model_openai,
                    api_key=self.openai_key,
                    temperature=self.html_temperature,
                    max_tokens=self.html_max_tokens,
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
                print(f"[TRACE][HTML_CONVERT] openai_error | msg={self._safe_str(e)}")

        # Gemini
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
                print(f"[TRACE][HTML_CONVERT] gemini_error | msg={self._safe_str(e)}")

        # remove ``` fences if model broke rules
        out = self._strip_code_fences(out).strip()
        print(f"[TRACE][HTML_CONVERT] done | output_chars={len(out)}")
        return out

    def _log_gemini_response_limits(self, resp: Any) -> None:
        """Log finish_reason / blocks so truncation vs safety is visible in traces."""
        try:
            pf = getattr(resp, "prompt_feedback", None)
            if pf is not None:
                br = getattr(pf, "block_reason", None)
                if br is not None:
                    bs = getattr(br, "name", None) or str(br)
                    if bs and "UNSPECIFIED" not in str(bs).upper():
                        print(f"[TRACE][HTML_CONVERT] gemini_pdf | prompt_feedback.block_reason={bs}")
            cands = getattr(resp, "candidates", None) or []
            if not cands:
                print("[TRACE][HTML_CONVERT] gemini_pdf | candidates=empty")
                return
            c0 = cands[0]
            fr = getattr(c0, "finish_reason", None)
            fr_s = getattr(fr, "name", None) if fr is not None else None
            detail = (fr_s or str(fr) or "?").strip()
            print(f"[TRACE][HTML_CONVERT] gemini_pdf | finish_reason={detail}")
            d = detail.upper()
            if "MAX" in d and "TOKEN" in d:
                print(
                    "⚠️ Gemini PDF HTML hit max_output_tokens this round. "
                    f"If continuations are enabled, another round may follow "
                    f"(HTML_MAX_TOKENS_PDF={self.html_max_tokens_pdf})."
                )
        except Exception as e:
            print(f"[TRACE][HTML_CONVERT] gemini_pdf | finish_diag_failed | {self._safe_str(e)}")

    def _gemini_finish_reason_is_max_tokens(self, resp: Any) -> bool:
        try:
            cands = getattr(resp, "candidates", None) or []
            if not cands:
                return False
            fr = getattr(cands[0], "finish_reason", None)
            if fr is None:
                return False
            name = getattr(fr, "name", None)
            if isinstance(name, str) and "MAX_TOKEN" in name.upper():
                return True
            if str(fr).upper().find("MAX_TOKEN") >= 0:
                return True
            try:
                return int(fr) == 2
            except Exception:
                return False
        except Exception:
            return False

    def _strip_duplicate_html_shell_from_continuation(self, piece: str) -> str:
        """If continuation repeats <!DOCTYPE>/<html>/<head>/<body>, drop the shell."""
        t = (piece or "").strip()
        if not t:
            return ""
        low_head = t[:4000].lower()
        if "<body" not in low_head:
            return t
        idx = low_head.find("<body")
        close = t.find(">", idx)
        if close > idx:
            return t[close + 1 :].lstrip()
        return t

    def _gemini_pdf_continuation_prompt(self, tail: str) -> str:
        tail = (tail or "").strip()
        return f"""You were converting the attached PDF into ONE continuous HTML document. The model output length limit cut off your previous reply mid-stream.

Output rules:
- Emit ONLY new HTML that CONTINUES immediately after the cut-off. Do not repeat any text already present below.
- Do NOT start a second full document unless you must emit closing tags for elements left open at the cut.
- Prefer completing any broken tag from the tail, then continue until the end of the PDF or until stopped again.

Last part of your previous output (continue after this; do not duplicate):
<<<
{tail}
>>>

The same PDF is attached again for reference."""

    def _generate_gemini_pdf_html_accumulated(
        self,
        model: Any,
        initial_prompt: str,
        ready_file: Any,
        generation_config: Dict[str, Any],
    ) -> str:
        """First pass + continuation passes until STOP or limits. Uses only Gemini (Files API)."""
        max_rounds = max(1, self.gemini_pdf_max_continuation_rounds)
        tail_cap = max(1000, self.gemini_pdf_continuation_tail_chars)
        acc = ""
        rounds_used = 0
        for r in range(max_rounds):
            if r == 0:
                contents: List[Any] = [initial_prompt, ready_file]
            else:
                tail = acc[-tail_cap:] if len(acc) > tail_cap else acc
                cp = self._gemini_pdf_continuation_prompt(tail)
                contents = [cp, ready_file]
                print(f"[TRACE][HTML_CONVERT] gemini_pdf | continuation_round={r + 1}/{max_rounds}")

            resp = model.generate_content(
                contents,
                generation_config=generation_config,
            )
            piece = (getattr(resp, "text", "") or "").strip()
            if r > 0:
                piece = self._strip_duplicate_html_shell_from_continuation(piece)
            acc += piece
            self._log_gemini_response_limits(resp)
            rounds_used = r + 1

            if not self._gemini_finish_reason_is_max_tokens(resp):
                break
            if not piece:
                print("[TRACE][HTML_CONVERT] gemini_pdf | continuation got empty piece, stop")
                break

        print(
            f"[TRACE][HTML_CONVERT] gemini_pdf | rounds_used={rounds_used} "
            f"accumulated_chars={len(acc)}"
        )
        return acc

    def _gemini_upload_state_name(self, file_obj: Any) -> str:
        st = getattr(file_obj, "state", None)
        if st is None:
            return ""
        name = getattr(st, "name", None)
        if isinstance(name, str) and name:
            return name
        return str(st)

    def _poll_gemini_file_ready(self, file_obj: Any, timeout_sec: float = 300.0) -> Any:
        """Wait until uploaded Files API resource is ACTIVE (or equivalent)."""
        if genai is None:
            raise RuntimeError(_gemini_import_explain())
        deadline = time.time() + timeout_sec
        f = file_obj
        while time.time() < deadline:
            state = self._gemini_upload_state_name(f)
            if state == "ACTIVE":
                return f
            if state == "FAILED":
                raise RuntimeError("Gemini file upload failed during processing")
            try:
                f = genai.get_file(f.name)
            except Exception as e:
                raise RuntimeError(f"Gemini get_file failed: {self._safe_str(e)}") from e
            time.sleep(2.0)
        raise RuntimeError("Timed out waiting for Gemini uploaded file to become ACTIVE")

    def to_structured_html_from_pdf(self, pdf_path: str, file_title: Optional[str] = None) -> str:
        """
        Convert a PDF on disk to structured HTML via Gemini Files API (multimodal).

        Uses the same SYSTEM_PROMPT rules as to_structured_html; input is the PDF file (not pypdf text).
        Does not call OpenAI. Long PDFs: multiple Gemini rounds append output when finish_reason is MAX_TOKENS
        (see GEMINI_PDF_MAX_CONTINUATION_ROUNDS, HTML_MAX_TOKENS_PDF).

        Env: GEMINI_API_KEY or GOOGLE_API_KEY; GEMINI_MODEL_NAME (optional; overrides HTML_MODEL_GEMINI).
        """
        path = (pdf_path or "").strip()
        if not path or not os.path.isfile(path):
            print("[TRACE][HTML_CONVERT_PDF] skip_missing_file")
            return ""

        title = (file_title or "").strip()
        if not title:
            base = os.path.basename(path)
            title = os.path.splitext(base)[0].strip() or "Document"

        base_prompt = self.SYSTEM_PROMPT.format(title=title).strip()

        pdf_notice = (
            "\nIMPORTANT: The input attached to this request is a PDF file (binary document). "
            "Read all pages from this PDF and extract the text faithfully into HTML as specified below. "
            "Do not rely on any separate plain-text CONTENT block."
        )

        prompt = f"""{base_prompt}
{pdf_notice}

Additional constraints:
- Return full valid HTML document only.
- Keep legal hierarchy when present (chapter/section/article) using h1/h2/h3.
- If Markdown table syntax appears in extracted text, convert it into valid HTML table structure.

There is no separate CONTENT block in this message; use only the attached PDF file as the source.
""".strip()

        sz = os.path.getsize(path)
        print(
            "[TRACE][HTML_CONVERT] start "
            "| provider=gemini_pdf_file "
            "| input=pdf_file "
            f"| model={self.html_model_gemini} "
            f"| gemini_ready={bool(self.gemini_ready)} "
            f"| bytes={sz}"
        )

        if genai is None:
            raise RuntimeError(_gemini_import_explain())
        if not self.gemini_ready:
            raise RuntimeError(
                "Gemini is not configured. Set GEMINI_API_KEY or GOOGLE_API_KEY in .env"
            )

        uploaded = None
        try:
            uploaded = genai.upload_file(path, mime_type="application/pdf")
            ready_file = self._poll_gemini_file_ready(uploaded)

            model = genai.GenerativeModel(self.html_model_gemini)
            gen_cfg = {
                "temperature": self.html_temperature,
                "max_output_tokens": self.html_max_tokens_pdf,
            }
            print(
                f"[TRACE][HTML_CONVERT] gemini_pdf | max_output_tokens={self.html_max_tokens_pdf} "
                f"| max_continuation_rounds={self.gemini_pdf_max_continuation_rounds}"
            )
            out = self._generate_gemini_pdf_html_accumulated(
                model, prompt, ready_file, gen_cfg
            )
        except Exception as e:
            print(f"[TRACE][HTML_CONVERT] gemini_pdf_error | msg={self._safe_str(e)}")
            raise
        finally:
            if uploaded is not None:
                try:
                    genai.delete_file(uploaded.name)
                except Exception:
                    pass

        out = self._strip_code_fences(out).strip()
        print(f"[TRACE][HTML_CONVERT] done | output_chars={len(out)}")
        return out

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
        if is_table_meta or self._looks_like_html_table(content) or self._looks_like_markdown_table(content):
            txt = self._html_to_text(content)
            txt = self._canonical_table_to_markdown(txt)
            return txt.strip() if txt.strip() else content.strip()
        return None

    def _canonical_table_to_markdown(self, text: str) -> str:
        raw = (text or "").strip()
        if not raw:
            return ""
        if not self._RE_CANON_TABLE.search(raw):
            return raw

        first_line = raw.splitlines()[0].strip()
        schema_match = re.search(r"schema:\s*([^|]+)", first_line, flags=re.IGNORECASE)
        headers = [h.strip() for h in (schema_match.group(1) if schema_match else "").split("|") if h.strip()]
        if not headers:
            return raw

        row_blocks = re.findall(r"\[([^\]]+)\]", raw)
        if not row_blocks:
            return raw

        rows: List[List[str]] = []
        for rb in row_blocks:
            kvs = [p.strip() for p in rb.split(",") if p.strip()]
            values_map: Dict[str, str] = {}
            for kv in kvs:
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    values_map[k.strip()] = v.strip()
            row_vals = [values_map.get(h, "") for h in headers]
            rows.append(row_vals)

        if not rows:
            return raw

        md_lines = []
        md_lines.append("| " + " | ".join(headers) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for r in rows:
            md_lines.append("| " + " | ".join(r) + " |")
        return "\n".join(md_lines).strip()

    def _context_chunk_header(self, doc: Dict, index: int) -> str:
        """Title + immediate upper legislative context only (compact for LLM)."""
        meta = doc.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        title = (
            meta.get("original_name")
            or meta.get("filename")
            or meta.get("doc_name")
            or meta.get("doc_key")
            or f"مصدر {index}"
        )
        title = self._clean_source_name(self._safe_str(title)) or f"مصدر {index}"
        l1 = self._safe_str(meta.get("level_1") or "").strip()
        l2 = self._safe_str(meta.get("level_2") or meta.get("h2") or "").strip()
        art = self._safe_str(meta.get("article") or meta.get("h3") or "").strip()
        is_table = bool(meta.get("is_table"))
        upper = " > ".join(x for x in [l1, l2, art] if x).strip()
        lines = [f"عنوان المستند: {title}"]
        lines.append(f"is_table: {'true' if is_table else 'false'}")
        if upper:
            lines.append(upper)
        return "\n".join(lines)

    def _build_context(self, context_docs: List[Dict]) -> str:
        picked = self._best_docs(context_docs or [], self.max_docs_in_context)
        blocks: List[str] = []
        for i, doc in enumerate(picked, 1):
            raw_content = self._safe_str(doc.get("content", "")).strip()
            if not raw_content:
                continue
            chunk_text = self._canonical_table_to_markdown(raw_content).strip()
            header = self._context_chunk_header(doc, i)
            blocks.append(f"{header}\nالنص:\n{chunk_text}")
        return "\n\n--------------------\n\n".join(blocks).strip()

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
        ans = self._strip_sources_titles_marker((answer or "").strip())
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

    def _strip_sources_titles_marker(self, text: str) -> str:
        t = self._safe_str(text or "")
        t = re.sub(
            r"\n?\s*SOURCES_TITLES_JSON\s*:\s*\[[\s\S]*?\]\s*$",
            "",
            t,
            flags=re.IGNORECASE,
        )
        return t.rstrip()

    def _extract_answer_and_source_titles(self, raw_output: str) -> tuple[str, List[str]]:
        txt = self._safe_str(raw_output).strip()
        if not txt:
            return "", []

        m = re.search(r"SOURCES_TITLES_JSON\s*:\s*(\[[\s\S]*?\])\s*$", txt, flags=re.IGNORECASE)
        if not m:
            return txt, []

        payload = (m.group(1) or "").strip()
        answer = txt[: m.start()].rstrip()
        titles: List[str] = []
        try:
            parsed = json.loads(payload)
            if isinstance(parsed, list):
                seen = set()
                for it in parsed:
                    s = self._safe_str(it).strip()
                    if not s:
                        continue
                    k = s.lower()
                    if k in seen:
                        continue
                    seen.add(k)
                    titles.append(s)
        except Exception:
            pass
        return answer, titles

    # -----------------------------------------------------
    # ✅ Article direct return
    # -----------------------------------------------------
    def _is_article_request(self, text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return False
        return bool(self._RE_ARTICLE_WORD.search(t))

    def _is_exec_rules_query(self, text: str) -> bool:
        return bool(self._RE_EXEC_RULES_Q.search((text or "").strip()))

    def _chunk_sort_key(self, doc: Dict[str, Any]) -> Any:
        meta = doc.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        try:
            return int(meta.get("chunk_index") or 0)
        except Exception:
            return meta.get("chunk_index") or 0

    def _article_direct_extract_is_trivial(self, cut: str, article_phrase: str) -> bool:
        """If true, skip direct return and let the LLM use full structured context."""
        c = (cut or "").strip()
        if not c:
            return True
        ap = self.normalize_arabic((article_phrase or "").strip())
        ac = self.normalize_arabic(c)
        lines = [ln.strip() for ln in c.split("\n") if ln.strip()]
        if len(lines) <= 1:
            if len(ac) <= max(len(ap) + 60, 100):
                return True
        return False

    def _is_no_info_answer_text(self, text: str) -> bool:
        return bool(self._RE_NO_INFO_ANSWER.search((text or "").strip()))

    def _context_contains_requested_article(self, context_text: str, article_phrase: str) -> bool:
        if not context_text or not article_phrase:
            return False
        return bool(self._cut_single_article(context_text, article_phrase))

    def _extract_requested_article_only(self, rq: str, context_docs: List[Dict[str, Any]]) -> str:
        article_phrase = self._extract_article_phrase(rq)
        if not article_phrase:
            return ""
        docs = sorted(
            self._best_docs(context_docs or [], max_docs=80),
            key=self._chunk_sort_key,
        )
        parts: List[str] = []
        for d in docs:
            raw = self._safe_str(d.get("content") or "").strip()
            if not raw:
                continue
            txt = self._html_to_text(raw).strip()
            if txt:
                parts.append(txt)
        merged = "\n\n".join(parts).strip()
        if not merged:
            return ""
        cut = self._cut_single_article(merged, article_phrase)
        if not cut:
            return ""
        if self._article_direct_extract_is_trivial(cut, article_phrase):
            return ""
        if len(cut) > self.article_direct_max_chars:
            cut = cut[: self.article_direct_max_chars].rstrip() + "…"
        return cut.strip()

    # -----------------------------------------------------
    # ✅ build_retrieval_query_smart
    # -----------------------------------------------------
    def build_retrieval_query_smart(self, refined: Dict[str, Any], user_query: str = "") -> str:
        rq = (refined.get("refined_question") or user_query or "").strip()
        article_phrase = self._extract_article_phrase(rq)
        wants_rules = self._is_exec_rules_query(rq)
        base_query = self.build_retrieval_query(refined)

        if not article_phrase:
            return base_query

        if wants_rules:
            retrieval_query = f"{base_query} {article_phrase} نص القواعد التنفيذية"
        else:
            retrieval_query = f"{base_query} {article_phrase} نص المادة"

        return retrieval_query.strip()

    # -----------------------------------------------------
    # Gemini fallback answer
    # -----------------------------------------------------
    def _gemini_answer_from_context(
        self,
        refined_question: str,
        context_text: str,
        user_role: str,
        supplement: str = "",
    ) -> str:
        if not self.gemini_ready:
            return "عذراً، نظام الذكاء الاصطناعي غير متصل حالياً."

        article_phrase = self._extract_article_phrase(refined_question)
        regulation_term_block = ""
        if article_phrase:
            regulation_term_block = f"""
- إذا كان طلب المستخدم عن مصطلح تنظيمي محدد مثل {article_phrase}:
  - أعد المصطلح كما ورد نصًا في «النص الرسمي المعتمد» دون إعادة صياغة أو تطبيع.
  - لا تستبدل المسمى بمسمى آخر (مثل فصل/بند/فقرة) إلا إذا كان النص نفسه يستخدم ذلك.
  - إن لم يوجد {article_phrase} نصًا، اكتب حرفيًا: "عذراً، {article_phrase} غير متوفرة في الملفات المرفوعة حالياً."
""".strip()
        extra = (supplement or "").strip()
        extra_parts = [x for x in [regulation_term_block, extra] if x]
        extra_block = f"\n{chr(10).join(extra_parts)}\n" if extra_parts else ""

        prompt = f"""
أنت مساعد رسمي لجامعة الأمير سطام.
دور المستخدم: {user_role}

تعليمات إلزامية:
- الإجابة الحصرية من النص المعتمد فقط: لا تذكر أي حقيقة أو تعريفًا أو موعدًا أو إجراءً أو استثناءً لم يظهر صراحةً في «النص الرسمي المعتمد»؛ لا تُكمِل بالمعرفة العامة أو التخمين؛ إن لم يكفِ النص للإجابة قل حرفيًا: "عذراً، هذه المعلومة غير متوفرة في الملفات المرفوعة حالياً."
- تنسيق الإجابة: ابدأ بجملة جوابية مباشرة على السؤال؛ قسِّم المتن بفقرات قصيرة وفارغة بينها؛ للإجراءات أو الخطوات المتتابعة استخدم ترقيمًا (١- ٢-) أو نقاطًا واضحة؛ يجوز عنوان فرعي قصير قبل فقرة طويلة؛ تجنّب كتلة نص واحدة دون تقسيم.
- لا تذكر كلمات مثل (سياق، ملف، مستند، RAG).
- اكتب بالعربية الفصحى وبأسلوب رسمي.
- ممنوع إرجاع أي تاقات HTML أو أكواد.
- عند الإشارة إلى حكم ورد في النص، نوِّه إليه داخل الفقرات ذاكرًا المسمى الرسمي كما في النص (مثل: وفقًا للمادة (...)، ولا تُدرِج قسمًا بعنوان «المصادر» ولا قائمة مراجع في نهاية الإجابة.
- إذا كان المقطع المعتمد يحمل الوسم is_table: true فاعرض محتواه كجدول Markdown صحيح (صف عناوين + فاصل --- + صفوف) دون تحويله إلى سرد فقري.
- في آخر سطر فقط: أعد القائمة بصيغة JSON لعناوين المستندات المستخدمة من النص المعتمد بهذا الشكل الحرفي:
  SOURCES_TITLES_JSON: ["عنوان 1","عنوان 2"]
  - استخدم عناوين «عنوان المستند: ...» كما وردت في السياق فقط.
  - إذا لم تستخدم أي عنوان، أعد: SOURCES_TITLES_JSON: []
{extra_block}
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
        answer, titles = self._extract_answer_and_source_titles(out)
        self.last_answer_source_titles = titles
        return answer.strip() or "عذراً، لم أتمكن من استخراج إجابة من الملفات."

    # -----------------------------------------------------
    # ✅ generate_response (محسّن)
    # -----------------------------------------------------
    def generate_response(
        self,
        user_query: str,
        context_docs: Optional[List[Dict]] = None,
        user_role: str = "طالب",
        system_message: Optional[str] = None,
        refined_question: Optional[str] = None,
        previous_turn: Optional[Dict[str, str]] = None,
    ) -> str:
        self.last_answer_source_titles = []
        user_query = (user_query or "").strip()
        if not user_query:
            return "اكتب سؤالك من فضلك."

        greet = self._handle_greeting(user_query)
        if greet:
            return greet

        rq = (refined_question or user_query).strip()
        article_phrase = self._extract_article_phrase(rq)
        prev_user = self._safe_str((previous_turn or {}).get("user") or "").strip()
        prev_assistant = self._safe_str((previous_turn or {}).get("assistant") or "").strip()
        previous_turn_block = ""
        if prev_user and prev_assistant:
            previous_turn_block = f"""
سياق المحادثة السابق (آخر تبادل):
- المستخدم: {prev_user}
- المساعد: {prev_assistant}
""".strip()

        # Build context for model when deterministic article extraction is unavailable.
        context_text = self._build_context(context_docs or [])
        if not context_text:
            return "عذراً، هذه المعلومة غير متوفرة في الملفات المرفوعة حالياً."

        base_system = (system_message or "").strip()
        role_line = f"دور المستخدم: {user_role}".strip()
        rules = f"""
أنت مساعد رسمي لجامعة الأمير سطام.

{role_line}

تعليمات إلزامية:
1) استخدم فقط المعلومات الموجودة في "النص الرسمي المعتمد" أدناه.
2) كل مقطع يبدأ بعنوان المستند و"السياق التشريعي" (المستويات العليا) ثم النص — اقرأ المقاطع ذات الصلة قبل الإجابة.
4) ممنوع إدخال أي معلومات من خارج النص.
5) إذا لم تجد في النص ما يجيب، اكتب حرفيًا:
   "عذراً، هذه المعلومة غير متوفرة حالياً."
6) لا تذكر كلمات مثل (سياق، ملف، مستند، RAG).
7) اكتب بالعربية الفصحى وبأسلوب رسمي.
8) ممنوع إرجاع أي تاقات HTML أو أكواد.
9) عند الإشارة إلى نصوص وردت في «النص الرسمي المعتمد»، أدرِج الإحالة ضمن جمل الإجابة بأسلوب رسمي، ذاكرًا المسمى كما ورد (المادة، الفصل، البند، اللائحة، القواعد التنفيذية، وما شابه)، مثل: وفقًا للمادة (...)، أو بموجب الفصل (...)، دون اختلاق أرقام أو عناوين غير ظاهرة في النص.
10) لا تُضِف قسمًا بعنوان «المصادر» ولا قائمة مراجع أو ذكر «المقطع [n]» في نهاية الإجابة؛ اكتفِ بالإحالات داخل النص حسب البند 9.
11) نظِّم الإجابة لتسهيل القراءة: جملة افتتاحية تجيب عن السؤال مباشرة؛ فقرات قصيرة مع فاصل بينها؛ للإجراءات المتسلسلة استخدم ترقيمًا أو نقاطًا؛ عنوان فرعي قصير اختياري قبل فقرة مطوّلة؛ تجنّب كتلة نص واحدة طويلة بلا تقسيم.
12) في آخر سطر فقط: أعد القائمة بصيغة JSON لعناوين المستندات المستخدمة من النص المعتمد بهذا الشكل الحرفي:
    SOURCES_TITLES_JSON: ["عنوان 1","عنوان 2"]
    - استخدم عناوين «عنوان المستند: ...» كما وردت في السياق فقط.
    - إذا لم تستخدم أي عنوان، أعد: SOURCES_TITLES_JSON: []
13) إذا كان المقطع المعتمد يحمل الوسم is_table: true فاعرض محتواه كجدول Markdown صحيح (صف عناوين + فاصل --- + صفوف) دون تحويله إلى سرد فقري.
14) إذا كان طلب المستخدم عن مصطلح تنظيمي محدد مثل «المادة ...»:
    - أعد المصطلح كما ورد نصًا في «النص الرسمي المعتمد» دون إعادة صياغة أو تطبيع.
    - لا تستبدل المسمى بمسمى آخر (مثل فصل/بند/فقرة) إلا إذا كان النص نفسه يستخدم ذلك.
    - إذا كان الطلب عن «{article_phrase or 'المادة المطلوبة'}» ولم يوجد نصًا في السياق، اكتب حرفيًا:
      "عذراً، {article_phrase or 'المادة المطلوبة'} غير متوفرة في الملفات المرفوعة حالياً."

{previous_turn_block}

النص الرسمي المعتمد:
{context_text}
""".strip()

        final_system = (base_system + "\n\n" + rules).strip() if base_system else rules
        messages = [SystemMessage(content=final_system), HumanMessage(content=rq)]

        supplement = previous_turn_block

        if not self.llm:
            ans = self._gemini_answer_from_context(
                rq, context_text, user_role, supplement=supplement
            )
            if (
                article_phrase
                and self._is_no_info_answer_text(ans)
                and self._context_contains_requested_article(context_text, article_phrase)
            ):
                corrective = (
                    f'تصحيح إلزامي: «{article_phrase}» موجودة نصًا داخل «النص الرسمي المعتمد». '
                    "أجب من النص الموجود فقط، ولا تُرجِع عبارة عدم التوفر."
                )
                ans = self._gemini_answer_from_context(
                    rq, context_text, user_role, supplement=(supplement + "\n" + corrective).strip()
                )
            return self.append_sources_to_answer(ans, context_docs or [])

        try:
            llm = self.llm
            try:
                llm = llm.bind(temperature=self.answer_temperature)
            except Exception:
                pass

            response = llm.invoke(messages)
            text = self._safe_str(getattr(response, "content", "")).strip()
            text, titles = self._extract_answer_and_source_titles(text)
            self.last_answer_source_titles = titles
            if (
                article_phrase
                and self._is_no_info_answer_text(text)
                and self._context_contains_requested_article(context_text, article_phrase)
            ):
                corrective_system = (
                    final_system
                    + "\n\n"
                    + f"تصحيح إلزامي: «{article_phrase}» موجودة نصًا داخل «النص الرسمي المعتمد». "
                      "أجب من النص الموجود فقط، ولا تُرجِع عبارة عدم التوفر."
                )
                retry_resp = llm.invoke([SystemMessage(content=corrective_system), HumanMessage(content=rq)])
                retry_text = self._safe_str(getattr(retry_resp, "content", "")).strip()
                retry_text, retry_titles = self._extract_answer_and_source_titles(retry_text)
                if retry_text:
                    text = retry_text
                    self.last_answer_source_titles = retry_titles
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
                    text, titles = self._extract_answer_and_source_titles(text)
                    self.last_answer_source_titles = titles
                    text = text or "عذراً، لم أتمكن من استخراج إجابة من الملفات."
                    return self.append_sources_to_answer(text, context_docs or [])
                except Exception:
                    pass

            ans = self._gemini_answer_from_context(
                rq, context_text, user_role, supplement=supplement
            )
            if (
                article_phrase
                and self._is_no_info_answer_text(ans)
                and self._context_contains_requested_article(context_text, article_phrase)
            ):
                corrective = (
                    f'تصحيح إلزامي: «{article_phrase}» موجودة نصًا داخل «النص الرسمي المعتمد». '
                    "أجب من النص الموجود فقط، ولا تُرجِع عبارة عدم التوفر."
                )
                ans = self._gemini_answer_from_context(
                    rq, context_text, user_role, supplement=(supplement + "\n" + corrective).strip()
                )
            return self.append_sources_to_answer(ans, context_docs or [])

    def _iter_text_deltas(self, text: str, chunk_size: int = 24) -> Iterator[str]:
        clean = self._safe_str(text)
        if not clean:
            return
        for i in range(0, len(clean), chunk_size):
            yield clean[i : i + chunk_size]

    def generate_response_stream(
        self,
        user_query: str,
        context_docs: Optional[List[Dict]] = None,
        user_role: str = "طالب",
        system_message: Optional[str] = None,
        refined_question: Optional[str] = None,
        previous_turn: Optional[Dict[str, str]] = None,
    ) -> Iterator[str]:
        """
        Stream assistant text deltas for SSE.
        Uses provider streaming when available; otherwise falls back to chunking
        the final generated response.
        """
        self.last_answer_source_titles = []
        user_query = (user_query or "").strip()
        if not user_query:
            for delta in self._iter_text_deltas("اكتب سؤالك من فضلك."):
                yield delta
            return

        greet = self._handle_greeting(user_query)
        if greet:
            for delta in self._iter_text_deltas(greet):
                yield delta
            return

        rq = (refined_question or user_query).strip()
        article_phrase = self._extract_article_phrase(rq)
        prev_user = self._safe_str((previous_turn or {}).get("user") or "").strip()
        prev_assistant = self._safe_str((previous_turn or {}).get("assistant") or "").strip()
        previous_turn_block = ""
        if prev_user and prev_assistant:
            previous_turn_block = f"""
سياق المحادثة السابق (آخر تبادل):
- المستخدم: {prev_user}
- المساعد: {prev_assistant}
""".strip()

        context_text = self._build_context(context_docs or [])
        if not context_text:
            for delta in self._iter_text_deltas("عذراً، هذه المعلومة غير متوفرة في الملفات المرفوعة حالياً."):
                yield delta
            return
        if article_phrase and self._context_contains_requested_article(context_text, article_phrase):
            final = self.generate_response(
                user_query=user_query,
                context_docs=context_docs,
                user_role=user_role,
                system_message=system_message,
                refined_question=refined_question,
                previous_turn=previous_turn,
            )
            for delta in self._iter_text_deltas(final):
                yield delta
            return

        base_system = (system_message or "").strip()
        role_line = f"دور المستخدم: {user_role}".strip()
        rules = f"""
أنت مساعد رسمي لجامعة الأمير سطام.

{role_line}

تعليمات إلزامية:
1) استخدم فقط المعلومات الموجودة في "النص الرسمي المعتمد" أدناه.
2) كل مقطع يبدأ بعنوان المستند و"السياق التشريعي" (المستويات العليا) ثم النص — اقرأ المقاطع ذات الصلة قبل الإجابة.
4) ممنوع إدخال أي معلومات من خارج النص.
5) إذا لم تجد في النص ما يجيب، اكتب حرفيًا:
   "عذراً، هذه المعلومة غير متوفرة حالياً."
6) لا تذكر كلمات مثل (سياق، ملف، مستند، RAG).
7) اكتب بالعربية الفصحى وبأسلوب رسمي.
8) ممنوع إرجاع أي تاقات HTML أو أكواد.
9) عند الإشارة إلى نصوص وردت في «النص الرسمي المعتمد»، أدرِج الإحالة ضمن جمل الإجابة بأسلوب رسمي، ذاكرًا المسمى كما ورد (المادة، الفصل، البند، اللائحة، القواعد التنفيذية، وما شابه)، مثل: وفقًا للمادة (...)، أو بموجب الفصل (...)، دون اختلاق أرقام أو عناوين غير ظاهرة في النص.
10) لا تُضِف قسمًا بعنوان «المصادر» ولا قائمة مراجع أو ذكر «المقطع [n]» في نهاية الإجابة؛ اكتفِ بالإحالات داخل النص حسب البند 9.
11) نظِّم الإجابة لتسهيل القراءة: جملة افتتاحية تجيب عن السؤال مباشرة؛ فقرات قصيرة مع فاصل بينها؛ للإجراءات المتسلسلة استخدم ترقيمًا أو نقاطًا؛ عنوان فرعي قصير اختياري قبل فقرة مطوّلة؛ تجنّب كتلة نص واحدة طويلة بلا تقسيم.
12) في آخر سطر فقط: أعد القائمة بصيغة JSON لعناوين المستندات المستخدمة من النص المعتمد بهذا الشكل الحرفي:
    SOURCES_TITLES_JSON: ["عنوان 1","عنوان 2"]
    - استخدم عناوين «عنوان المستند: ...» كما وردت في السياق فقط.
    - إذا لم تستخدم أي عنوان، أعد: SOURCES_TITLES_JSON: []
13) إذا كان المقطع المعتمد يحمل الوسم is_table: true فاعرض محتواه كجدول Markdown صحيح (صف عناوين + فاصل --- + صفوف) دون تحويله إلى سرد فقري.
14) إذا كان طلب المستخدم عن مصطلح تنظيمي محدد مثل «المادة ...»:
    - أعد المصطلح كما ورد نصًا في «النص الرسمي المعتمد» دون إعادة صياغة أو تطبيع.
    - لا تستبدل المسمى بمسمى آخر (مثل فصل/بند/فقرة) إلا إذا كان النص نفسه يستخدم ذلك.
    - إذا كان الطلب عن «{article_phrase or 'المادة المطلوبة'}» ولم يوجد نصًا في السياق، اكتب حرفيًا:
      "عذراً، {article_phrase or 'المادة المطلوبة'} غير متوفرة في الملفات المرفوعة حالياً."

{previous_turn_block}

النص الرسمي المعتمد:
{context_text}
""".strip()

        final_system = (base_system + "\n\n" + rules).strip() if base_system else rules
        messages = [SystemMessage(content=final_system), HumanMessage(content=rq)]
        supplement = previous_turn_block

        if not self.llm:
            ans = self._gemini_answer_from_context(
                rq, context_text, user_role, supplement=supplement
            )
            final = self.append_sources_to_answer(ans, context_docs or [])
            for delta in self._iter_text_deltas(final):
                yield delta
            return

        streamed_any = False
        parts: List[str] = []
        try:
            llm = self.llm
            try:
                llm = llm.bind(temperature=self.answer_temperature)
            except Exception:
                pass

            marker = "SOURCES_TITLES_JSON"
            pending = ""
            marker_found = False
            keep_tail = len(marker) + 6
            for chunk in llm.stream(messages):
                piece = self._safe_str(getattr(chunk, "content", ""))
                if not piece:
                    continue
                streamed_any = True
                parts.append(piece)
                if marker_found:
                    continue
                pending += piece
                if marker in pending:
                    safe_part = pending.split(marker, 1)[0]
                    safe_part = self._strip_sources_titles_marker(safe_part)
                    if safe_part:
                        yield safe_part
                    marker_found = True
                    pending = ""
                    continue
                if len(pending) > keep_tail:
                    out = pending[:-keep_tail]
                    pending = pending[-keep_tail:]
                    if out:
                        yield out
            if not marker_found and pending:
                final_pending = self._strip_sources_titles_marker(pending)
                if final_pending:
                    yield final_pending
        except Exception:
            streamed_any = False

        if streamed_any:
            combined = self._safe_str("".join(parts)).strip()
            _, titles = self._extract_answer_and_source_titles(combined)
            self.last_answer_source_titles = titles
            return

        # Safe fallback when provider streaming is unavailable.
        final = self.generate_response(
            user_query=user_query,
            context_docs=context_docs,
            user_role=user_role,
            system_message=system_message,
            refined_question=refined_question,
            previous_turn=previous_turn,
        )
        for delta in self._iter_text_deltas(final):
            yield delta

    def is_available(self) -> bool:
        return (self.llm is not None) or bool(self.gemini_ready)