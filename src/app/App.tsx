import { useEffect, useRef, useState } from "react";
import React from "react";
import { BrowserRouter, Routes, Route, Navigate, useLocation } from "react-router-dom";

import { LandingPage } from "./pages/LandingPage";
import { ChatPage } from "./pages/ChatPage";

import { AdminLogin } from "./pages/admin/AdminLogin";
import DocumentsPage from "./pages/admin/DocumentsPage";

import { apiGet } from "../api";

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const location = useLocation();
  const [loading, setLoading] = useState(true);
  const [allowed, setAllowed] = useState(false);

  const didCheckRef = useRef(false);

  useEffect(() => {
    if (didCheckRef.current) return;
    didCheckRef.current = true;

    const token = sessionStorage.getItem("adminAuth");

    if (!token) {
      setAllowed(false);
      setLoading(false);
      return;
    }

    apiGet("/api/admin/stats")
      .then(() => setAllowed(true))
      .catch(() => {
        sessionStorage.removeItem("adminAuth");
        setAllowed(false);
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center" dir="rtl">
        جاري التحقق من صلاحيات الإدارة...
      </div>
    );
  }

  return allowed ? <>{children}</> : <Navigate to="/admin/login" replace state={{ from: location }} />;
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/chat" element={<ChatPage />} />

        <Route path="/admin/login" element={<AdminLogin />} />

        {/* ✅ الأدمن صار صفحة وحدة فقط: إدارة الملفات */}
        <Route
          path="/admin"
          element={
            <ProtectedRoute>
              <DocumentsPage />
            </ProtectedRoute>
          }
        />

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
