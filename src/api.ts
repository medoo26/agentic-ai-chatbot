const DEFAULT_BASE_URL = "http://127.0.0.1:8000";

export const API_BASE_URL =
  (import.meta as any).env?.VITE_API_BASE_URL?.trim() || DEFAULT_BASE_URL;

type Json = Record<string, any>;
const CHAT_SESSION_STORAGE_KEY = "chatSessionId";

/* --------------------------------------------------
   Helpers
-------------------------------------------------- */

/**
 * جلب بيانات الاعتماد الخاصة بالأدمن من التخزين
 */
function getAdminAuthHeader(): Record<string, string> {
  const token = sessionStorage.getItem("adminAuth");
  if (!token) return {};
  return { Authorization: `Basic ${token}` };
}

/**
 * بناء الرؤوس (بدون Session/Cookies)
 */
function buildHeaders(path: string, extra?: Record<string, string>): HeadersInit {
  const headers: Record<string, string> = {
    Accept: "application/json",
    ...(extra || {}),
  };

  // admin endpoints تحتاج Authorization
  if (path.startsWith("/api/admin")) {
    Object.assign(headers, getAdminAuthHeader());
  }

  return headers;
}

export function getOrCreateChatSessionId(): string {
  const existing = localStorage.getItem(CHAT_SESSION_STORAGE_KEY)?.trim();
  if (existing) return existing;
  const generated =
    (globalThis.crypto as any)?.randomUUID?.()?.replace(/-/g, "") ||
    `${Date.now()}${Math.random().toString(36).slice(2, 10)}`;
  localStorage.setItem(CHAT_SESSION_STORAGE_KEY, generated);
  return generated;
}

/**
 * معالجة الرد
 */
async function handleResponse(res: Response) {
  const text = await res.text();
  let data: any = null;

  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text || null;
  }

  if (!res.ok) {
    const msg =
      (data && (data.detail || data.message)) || `Request failed (${res.status})`;
    throw new Error(msg);
  }

  return data;
}

/* --------------------------------------------------
   Generic API Methods
-------------------------------------------------- */

export async function apiGet<T = any>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    method: "GET",
    headers: buildHeaders(path),
    credentials: "omit", // ✅ بدون كوكيز
  });
  return handleResponse(res) as Promise<T>;
}

export async function apiPost<T = any>(path: string, body?: Json): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: buildHeaders(path, { "Content-Type": "application/json" }),
    body: body ? JSON.stringify(body) : undefined,
    credentials: "omit", // ✅ بدون كوكيز
  });
  return handleResponse(res) as Promise<T>;
}

/* --------------------------------------------------
   Upload (FormData)
-------------------------------------------------- */

export type ChatStreamMeta = {
  id: number;
  timestamp: string;
  sources?: Array<{ name: string; page?: number | null }>;
  attachments?: Array<{ name: string; url: string; mime?: string; size_mb?: number }>;
  choices?: Array<{ label: string; doc_key: string }>;
  debug?: string | null;
};

type ChatStreamOptions = {
  signal?: AbortSignal;
  onToken?: (delta: string) => void;
  onMeta?: (meta: ChatStreamMeta) => void;
  onDone?: () => void;
  onError?: (message: string) => void;
};

export async function apiPostChatStream(
  path: string,
  body: Json,
  opts: ChatStreamOptions = {}
): Promise<void> {
  const sessionId = getOrCreateChatSessionId();
  const res = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: buildHeaders(path, {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
      "X-Chat-Session-Id": sessionId,
    }),
    body: JSON.stringify({ ...(body || {}), session_id: sessionId }),
    credentials: "omit",
    signal: opts.signal,
  });

  if (!res.ok || !res.body) {
    const text = await res.text().catch(() => "");
    let msg = `Request failed (${res.status})`;
    try {
      const data = text ? JSON.parse(text) : null;
      msg = (data && (data.detail || data.message)) || msg;
    } catch {
      if (text) msg = text;
    }
    throw new Error(msg);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  const processEvent = (rawEvent: string) => {
    const lines = rawEvent.split("\n");
    let eventName = "message";
    const dataLines: string[] = [];

    for (const line of lines) {
      if (line.startsWith("event:")) {
        eventName = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trim());
      }
    }

    if (!dataLines.length) return;
    let payload: any = null;
    try {
      payload = JSON.parse(dataLines.join("\n"));
    } catch {
      payload = { message: dataLines.join("\n") };
    }

    if (eventName === "token") {
      opts.onToken?.(String(payload?.delta ?? ""));
    } else if (eventName === "meta") {
      opts.onMeta?.(payload as ChatStreamMeta);
    } else if (eventName === "done") {
      opts.onDone?.();
    } else if (eventName === "error") {
      opts.onError?.(String(payload?.message ?? "Streaming failed"));
    }
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const chunks = buffer.split("\n\n");
      buffer = chunks.pop() || "";
      chunks.forEach(processEvent);
    }
    if (buffer.trim()) processEvent(buffer.trim());
  } finally {
    reader.releaseLock();
  }
}
