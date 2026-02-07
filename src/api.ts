const DEFAULT_BASE_URL = "http://127.0.0.1:8000";

export const API_BASE_URL =
  (import.meta as any).env?.VITE_API_BASE_URL?.trim() || DEFAULT_BASE_URL;

type Json = Record<string, any>;

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

export async function apiPut<T = any>(path: string, body?: Json): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    method: "PUT",
    headers: buildHeaders(path, { "Content-Type": "application/json" }),
    body: body ? JSON.stringify(body) : undefined,
    credentials: "omit", // ✅ بدون كوكيز
  });
  return handleResponse(res) as Promise<T>;
}

export async function apiDelete<T = any>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    method: "DELETE",
    headers: buildHeaders(path),
    credentials: "omit", // ✅ بدون كوكيز
  });
  return handleResponse(res) as Promise<T>;
}

/* --------------------------------------------------
   Upload (FormData)
-------------------------------------------------- */

export async function apiPostForm<T = any>(path: string, form: FormData): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: buildHeaders(path),
    body: form,
    credentials: "omit", // ✅ بدون كوكيز
  });
  return handleResponse(res) as Promise<T>;
}
