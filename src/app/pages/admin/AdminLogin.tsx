import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "../../components/ui/button";
import { Input } from "../../components/ui/input";
import { Card } from "../../components/ui/card";
import { Lock, User, GraduationCap } from "lucide-react";
import { apiGet } from "../../../api";

export function AdminLogin() {
  const navigate = useNavigate();

  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("admin123");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      // خزّن Basic token
      const token = btoa(`${username}:${password}`);
      sessionStorage.setItem("adminAuth", token);

      // تحقق فعلي من الباكند (لو غلط بيرجع 401)
      await apiGet("/api/admin/stats");

      // ✅ نجاح -> روح للإدارة الرئيسية (بدل /admin/dashboard)
      navigate("/admin", { replace: true });
    } catch (err: any) {
      sessionStorage.removeItem("adminAuth");
      setError(err?.message || "فشل تسجيل الدخول");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      className="min-h-screen bg-gradient-to-b from-white to-[#E8F5E9] flex items-center justify-center p-6"
      dir="rtl"
    >
      <Card className="w-full max-w-md p-8 space-y-6 bg-white shadow-xl">
        <div className="text-center space-y-4">
          <div className="flex justify-center">
            <div className="w-20 h-20 bg-[#2E7D32] rounded-full flex items-center justify-center">
              <GraduationCap className="w-10 h-10 text-white" />
            </div>
          </div>
          <h1 className="text-2xl font-bold text-[#2E7D32]">لوحة تحكم الإدارة</h1>
          <p className="text-gray-600">تسجيل الدخول للوصول إلى لوحة التحكم</p>
        </div>

        <form onSubmit={handleLogin} className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm text-gray-700">اسم المستخدم</label>
            <div className="relative">
              <User className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 w-5 h-5" />
              <Input
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="admin"
                className="pr-10 rounded-xl"
                required
                autoComplete="username"
              />
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-sm text-gray-700">كلمة المرور</label>
            <div className="relative">
              <Lock className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 w-5 h-5" />
              <Input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="pr-10 rounded-xl"
                required
                autoComplete="current-password"
              />
            </div>
          </div>

          {error && (
            <div className="text-sm text-red-600 bg-red-50 border border-red-200 p-3 rounded-xl">
              {error}
            </div>
          )}

          <Button
            type="submit"
            disabled={loading}
            className="w-full bg-[#2E7D32] hover:bg-[#1B5E20] text-white rounded-xl py-6"
          >
            {loading ? "جاري التحقق..." : "تسجيل الدخول"}
          </Button>
        </form>

        <div className="text-center pt-4">
          <button
            type="button"
            onClick={() => navigate("/")}
            className="text-sm text-[#2E7D32] hover:underline"
          >
            العودة للصفحة الرئيسية
          </button>
        </div>
      </Card>
    </div>
  );
}
