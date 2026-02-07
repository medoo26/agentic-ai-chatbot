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
import { Upload, Trash2, Loader2, Paperclip, FileText, LogOut } from "lucide-react";

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

export default function DocumentsPage() {
  const navigate = useNavigate();

  const [docs, setDocs] = useState<DocumentOut[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);

  const [file, setFile] = useState<File | null>(null);

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
    }, 3000);
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

  const uploadDoc = async () => {
    if (!file) {
      setErr("اختر ملف أولاً");
      return;
    }

    setBusy(true);
    setErr("");
    setMsg("");

    try {
      const token = sessionStorage.getItem("adminAuth") || "";

      const fd = new FormData();
      fd.append("file", file);

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

      setMsg("تم رفع الملف ✅");
      setFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
      await loadDocs();
    } catch (e: any) {
      setErr(e?.message || "فشل رفع الملف");
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

  return (
    <div className="min-h-screen bg-gray-50" dir="rtl">
      {/* ✅ Top Bar (بدل AdminDashboard) */}
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

      {/* ✅ محتوى الصفحة */}
      <main className="p-6">
        <div className="space-y-6">
          {/* Toast */}
          {(err || msg) && (
            <div className="fixed top-4 right-4 z-50">
              <div
                className={`min-w-[260px] max-w-[360px] rounded-xl border px-4 py-3 shadow-lg text-sm flex items-start gap-3 ${
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

          {/* ✅ Summary Card */}
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
                accept=".pdf,.doc,.docx,.txt"
                className="hidden"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
              />

              <div className="flex flex-col gap-2 flex-1 w-full">
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="w-full h-11 px-4 rounded-xl border border-gray-200 bg-white hover:bg-gray-50 flex items-center justify-between gap-3"
                >
                  <span className="flex items-center gap-2 text-gray-700">
                    <Paperclip className="w-5 h-5 text-gray-500" />
                    {file ? "تم اختيار ملف" : "اختر ملف"}
                  </span>
                  <span className="text-xs text-gray-500 truncate max-w-[260px]">
                    {file ? file.name : "PDF / DOCX / TXT"}
                  </span>
                </button>

                {file && (
                  <div className="text-xs text-gray-600 bg-gray-50 border border-gray-200 rounded-xl px-3 py-2">
                    <span className="font-semibold">اسم الملف:</span>{" "}
                    <span className="break-all">{file.name}</span>
                  </div>
                )}
              </div>

              <Button
                onClick={uploadDoc}
                disabled={!file || busy}
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
                onClick={() => {
                  setFile(null);
                  if (fileInputRef.current) fileInputRef.current.value = "";
                }}
                disabled={!file || busy}
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
