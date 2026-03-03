import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

import { Button } from "../../components/ui/button";
import { Card } from "../../components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../../components/ui/table";
import { apiDelete, apiGet } from "../../../api";
import {
  Upload,
  Trash2,
  Loader2,
  Paperclip,
  FileText,
  LogOut,
} from "lucide-react";

type DocumentOut = {
  id: number;
  name: string;
  size_mb?: number | null;
  uploaded_at?: string | null;
  is_active: boolean;
};

const API_BASE = "http://127.0.0.1:8000";

function formatDateAr(iso?: string | null) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleDateString("ar-SA");
}

function formatSize(mb?: number | null) {
  if (mb == null) return "—";
  return `${mb.toFixed(2)} MB`;
}

type UploadResultItem = {
  filename: string;
  status: "success" | "exists" | "error";
  message?: string;
};

type UploadResponse = {
  status: "done";
  count: number;
  results: UploadResultItem[];
};

export default function DocumentsPage() {
  const navigate = useNavigate();

  const [docs, setDocs] = useState<DocumentOut[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);

  // ✅ multiple
  const [files, setFiles] = useState<File[]>([]);

  const [msg, setMsg] = useState("");
  const [err, setErr] = useState("");

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // ✅ Summary
  const totalFiles = docs.length;
  const totalSizeMb = docs.reduce((sum, d) => {
    const v = typeof d.size_mb === "number" ? d.size_mb : 0;
    return sum + v;
  }, 0);

  const handleLogout = () => {
    sessionStorage.removeItem("adminAuth");
    navigate("/admin/login");
  };

  // Toast auto hide
  useEffect(() => {
    if (!err && !msg) return;
    const t = setTimeout(() => {
      setErr("");
      setMsg("");
    }, 3500);
    return () => clearTimeout(t);
  }, [err, msg]);

  const loadDocs = async () => {
    setLoading(true);
    setErr("");
    try {
      const data = await apiGet("/api/admin/documents");
      setDocs(Array.isArray(data) ? data : []);
    } catch (e: any) {
      setErr(e?.message || "تعذر تحميل الملفات");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDocs();
  }, []);

  const clearSelected = () => {
    setFiles([]);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const onPickFiles = (list?: FileList | null) => {
    const picked = list ? Array.from(list) : [];
    setFiles(picked);
  };

  const summarizeUpload = (results: UploadResultItem[]) => {
    const ok = results.filter((r) => r.status === "success").length;
    const exists = results.filter((r) => r.status === "exists").length;
    const bad = results.filter((r) => r.status === "error").length;

    const firstError =
      results.find((r) => r.status === "error")?.message ||
      results.find((r) => r.status === "error")?.filename;

    if (bad > 0) {
      const tail = firstError ? `\nأول خطأ: ${firstError}` : "";
      return {
        isError: true,
        text: `تمت عملية الرفع مع أخطاء.\nنجح: ${ok} | موجود مسبقًا: ${exists} | فشل: ${bad}${tail}`,
      };
    }

    return {
      isError: false,
      text: `تم رفع الملفات ✅\nنجح: ${ok} | موجود مسبقًا: ${exists}`,
    };
  };

  const uploadDocs = async () => {
    if (!files || files.length === 0) {
      setErr("اختر ملف/ملفات أولاً");
      return;
    }

    setBusy(true);
    setErr("");
    setMsg("");

    try {
      const token = sessionStorage.getItem("adminAuth") || "";

      const fd = new FormData();
      // ✅ IMPORTANT: اسم الحقل لازم يكون "files"
      files.forEach((f) => fd.append("files", f));

      const res = await fetch(`${API_BASE}/api/admin/documents/upload`, {
        method: "POST",
        headers: {
          Accept: "application/json",
          ...(token ? { Authorization: `Basic ${token}` } : {}),
        },
        body: fd,
      });

      const text = await res.text();
      if (!res.ok) throw new Error(text || `Upload failed (${res.status})`);

      let data: UploadResponse | null = null;
      try {
        data = JSON.parse(text);
      } catch {
        data = null;
      }

      if (data?.results && Array.isArray(data.results)) {
        const summary = summarizeUpload(data.results);
        if (summary.isError) setErr(summary.text);
        else setMsg(summary.text);
      } else {
        setMsg("تم رفع الملفات ✅");
      }

      clearSelected();
      await loadDocs();
    } catch (e: any) {
      setErr(e?.message || "فشل رفع الملفات");
    } finally {
      setBusy(false);
    }
  };

  const deleteDoc = async (id: number) => {
    if (!confirm("حذف الملف؟")) return;

    setBusy(true);
    setErr("");
    setMsg("");

    try {
      await apiDelete(`/api/admin/documents/${id}`);
      setMsg("تم حذف الملف ✅");
      await loadDocs();
    } catch (e: any) {
      setErr(e?.message || "فشل حذف الملف");
    } finally {
      setBusy(false);
    }
  };

  const selectedLabel =
    files.length === 0
      ? "اختر ملفات"
      : files.length === 1
      ? "تم اختيار ملف"
      : `تم اختيار ${files.length} ملفات`;

  return (
    <div className="min-h-screen bg-gray-50" dir="rtl">
      {/* ✅ Top Bar */}
      <div className="sticky top-0 z-40 bg-white border-b border-gray-200">
        <div className="h-14 px-4 flex items-center justify-between">
          <div className="font-bold text-gray-800">إدارة الملفات</div>

          <Button
            onClick={handleLogout}
            variant="outline"
            className="text-red-600 border-red-200 hover:bg-red-50"
            style={{ minHeight: 44 }}
          >
            <LogOut className="ml-2 h-4 w-4" />
            خروج
          </Button>
        </div>
      </div>

      <main className="p-6">
        <div className="space-y-6">
          {/* Toast */}
          {(err || msg) && (
            <div className="fixed top-4 right-4 z-50 whitespace-pre-line">
              <div
                className={`min-w-[260px] max-w-[420px] rounded-xl border px-4 py-3 shadow-lg text-sm flex items-start gap-3 ${
                  err
                    ? "bg-red-50 border-red-200 text-red-700"
                    : "bg-green-50 border-green-200 text-green-700"
                }`}
              >
                <div className="pt-0.5">{err ? "❌" : "✅"}</div>
                <div className="flex-1">
                  <div className="font-semibold mb-0.5">
                    {err ? "صار خطأ" : "تم بنجاح"}
                  </div>
                  <div className="leading-5">{err || msg}</div>
                </div>
                <button
                  onClick={() => {
                    setErr("");
                    setMsg("");
                  }}
                  className="text-gray-500 hover:text-gray-700"
                >
                  ✕
                </button>
              </div>
            </div>
          )}

          {/* Summary */}
          <Card className="p-4 rounded-2xl">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-8">
                <div>
                  <div className="text-sm text-gray-500">إجمالي الملفات</div>
                  <div className="text-2xl font-bold">{totalFiles}</div>
                </div>

                <div className="h-10 w-px bg-gray-200 hidden sm:block" />

                <div>
                  <div className="text-sm text-gray-500">إجمالي الحجم</div>
                  <div className="text-2xl font-bold">
                    {totalSizeMb.toFixed(2)} MB
                  </div>
                </div>
              </div>

              <div className="w-12 h-12 rounded-2xl bg-green-50 flex items-center justify-center">
                <FileText className="w-6 h-6 text-[#2E7D32]" />
              </div>
            </div>
          </Card>

          {/* Upload */}
          <Card className="p-4">
            <div className="flex flex-col md:flex-row gap-3 items-start md:items-center">
              <input
                ref={fileInputRef}
                type="file"
                multiple // ✅
                accept=".pdf,.doc,.docx,.txt"
                className="hidden"
                onChange={(e) => onPickFiles(e.target.files)}
              />

              <div className="flex flex-col gap-2 flex-1 w-full">
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="w-full h-11 px-4 rounded-xl border border-gray-200 bg-white hover:bg-gray-50 flex items-center justify-between gap-3"
                >
                  <span className="flex items-center gap-2 text-gray-700">
                    <Paperclip className="w-5 h-5 text-gray-500" />
                    {selectedLabel}
                  </span>
                  <span className="text-xs text-gray-500 truncate max-w-[280px]">
                    {files.length === 0
                      ? "PDF / DOCX / TXT"
                      : files.length === 1
                      ? files[0].name
                      : `${files.length} ملفات`}
                  </span>
                </button>

                {files.length > 0 && (
                  <div className="text-xs text-gray-600 bg-gray-50 border border-gray-200 rounded-xl px-3 py-2 space-y-1">
                    <div className="font-semibold">الملفات المختارة:</div>
                    <div className="max-h-24 overflow-auto pr-1 space-y-0.5">
                      {files.map((f) => (
                        <div key={`${f.name}-${f.size}-${f.lastModified}`} className="break-all">
                          • {f.name}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              <Button
                onClick={uploadDocs}
                disabled={files.length === 0 || busy}
                className="bg-[#2E7D32] hover:bg-[#1B5E20] rounded-xl w-full md:w-auto"
              >
                {busy ? (
                  <Loader2 className="ml-2 h-4 w-4 animate-spin" />
                ) : (
                  <Upload className="ml-2 h-4 w-4" />
                )}
                رفع
              </Button>

              <Button
                type="button"
                variant="outline"
                className="rounded-xl w-full md:w-auto"
                onClick={clearSelected}
                disabled={files.length === 0 || busy}
              >
                إلغاء
              </Button>
            </div>
          </Card>

          {/* Table */}
          <Card className="p-4">
            {loading ? (
              "جاري التحميل..."
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>الاسم</TableHead>
                    <TableHead>الحجم</TableHead>
                    <TableHead>التاريخ</TableHead>
                    <TableHead>إجراءات</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {docs.map((d) => (
                    <TableRow key={d.id}>
                      <TableCell className="font-medium">{d.name}</TableCell>
                      <TableCell>{formatSize(d.size_mb)}</TableCell>
                      <TableCell>{formatDateAr(d.uploaded_at)}</TableCell>
                      <TableCell>
                        <Button
                          size="sm"
                          variant="outline"
                          className="text-red-600 border-red-200 hover:bg-red-50 rounded-xl"
                          onClick={() => deleteDoc(d.id)}
                          disabled={busy}
                        >
                          <Trash2 className="ml-1 h-4 w-4" />
                          حذف
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}

                  {docs.length === 0 && (
                    <TableRow>
                      <TableCell colSpan={4} className="text-center text-gray-500">
                        لا توجد ملفات
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            )}
          </Card>
        </div>
      </main>
    </div>
  );
}