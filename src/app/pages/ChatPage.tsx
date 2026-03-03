import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { ScrollArea } from "../components/ui/scroll-area";
import { Send, LogOut, Plus, Bot, Download, FileText } from "lucide-react";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// ✅ API
import { apiPost, API_BASE_URL } from "../../api";

type Attachment = {
  name: string;
  url: string; // غالباً تجي "/api/files/123"
  mime?: string;
  size_mb?: number;
};

type ChoiceItem = {
  label: string;   // اسم الملف الظاهر
  doc_key: string; // doc_key الحقيقي للفلترة
};

type ApiMessage = {
  id: number;
  content: string;
  sender: "user" | "assistant";
  timestamp: string;
  sources?: string[];
  attachments?: Attachment[];

  // ✅ NEW: خيارات اختيار المصدر
  choices?: ChoiceItem[];
};

type UIMessage = {
  id: string;
  content: string;
  sender: "user" | "assistant";
  timestamp: Date;
  sources?: string[];
  attachments?: Attachment[];
  choices?: ChoiceItem[];
};

export function ChatPage() {
  const navigate = useNavigate();

  const [messages, setMessages] = useState<UIMessage[]>([]);
  const [inputMessage, setInputMessage] = useState("");
  const [sending, setSending] = useState(false);

  // ✅ لمنع سبام الضغط على أكثر من خيار بسرعة
  const [choosing, setChoosing] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages.length]);

  const handleLogout = () => navigate("/");

  const createNewChat = () => {
    setMessages([]);
    setInputMessage("");
  };

  // ✅ Helper لبناء رابط كامل للتحميل
  const toAbsoluteUrl = (url: string) => {
    if (!url) return "";
    if (/^https?:\/\//i.test(url)) return url;
    return `${API_BASE_URL}${url.startsWith("/") ? url : `/${url}`}`;
  };

  // ✅ إرسال رسالة عادية
  const sendMessage = async (content: string) => {
    if (!content.trim() || sending || choosing) return;
    setSending(true);

    const userMsg: UIMessage = {
      id: `u-${Date.now()}`,
      content,
      sender: "user",
      timestamp: new Date(),
      sources: [],
      attachments: [],
      choices: [],
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputMessage("");

    try {
      const ai = await apiPost<ApiMessage>("/api/chat", { content });

      const aiMsg: UIMessage = {
        id: `a-${ai.id}-${Date.now()}`,
        content: ai.content,
        sender: ai.sender,
        timestamp: new Date(ai.timestamp),
        sources: ai.sources || [],
        attachments: Array.isArray(ai.attachments) ? ai.attachments : [],
        choices: Array.isArray(ai.choices) ? ai.choices : [],
      };

      setMessages((prev) => [...prev, aiMsg]);
    } catch (err: any) {
      const errMsg: UIMessage = {
        id: `err-${Date.now()}`,
        content: `صار خطأ: ${err?.message || "غير معروف"}`,
        sender: "assistant",
        timestamp: new Date(),
        sources: [],
        attachments: [],
        choices: [],
      };
      setMessages((prev) => [...prev, errMsg]);
    } finally {
      setSending(false);
    }
  };

  // ✅ إرسال اختيار مصدر (زر)
  // الفكرة: نرسل content للعرض + choice_doc_key للباكند
  const sendChoice = async (label: string, doc_key: string) => {
    if (!label || !doc_key || sending || choosing) return;

    setChoosing(true);

    // (اختياري) نعرض اختيارك كفقاعة مستخدم صغيرة
    const userMsg: UIMessage = {
      id: `u-choice-${Date.now()}`,
      content: label,
      sender: "user",
      timestamp: new Date(),
      sources: [],
      attachments: [],
      choices: [],
    };
    setMessages((prev) => [...prev, userMsg]);

    try {
      const ai = await apiPost<ApiMessage>("/api/chat", {
        content: label,
        choice_doc_key: doc_key, // ✅ مهم
      });

      const aiMsg: UIMessage = {
        id: `a-${ai.id}-${Date.now()}`,
        content: ai.content,
        sender: ai.sender,
        timestamp: new Date(ai.timestamp),
        sources: ai.sources || [],
        attachments: Array.isArray(ai.attachments) ? ai.attachments : [],
        choices: Array.isArray(ai.choices) ? ai.choices : [],
      };

      setMessages((prev) => [...prev, aiMsg]);
    } catch (err: any) {
      const errMsg: UIMessage = {
        id: `err-${Date.now()}`,
        content: `صار خطأ: ${err?.message || "غير معروف"}`,
        sender: "assistant",
        timestamp: new Date(),
        sources: [],
        attachments: [],
        choices: [],
      };
      setMessages((prev) => [...prev, errMsg]);
    } finally {
      setChoosing(false);
    }
  };

  return (
    <div className="h-dvh flex flex-col bg-white overflow-hidden" dir="rtl">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 p-3 lg:p-4 shadow-sm">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Button
              onClick={handleLogout}
              variant="outline"
              className="h-11 px-4 border-gray-300 text-gray-700 hover:bg-gray-50 rounded-xl"
              title="خروج"
            >
              <LogOut className="w-4 h-4 ml-2" />
              خروج
            </Button>

            <Button
              onClick={createNewChat}
              variant="outline"
              className="h-11 px-4 border-[#2E7D32] text-[#2E7D32] hover:bg-[#E8F5E9] rounded-xl"
              title="محادثة جديدة"
            >
              <Plus className="w-4 h-4 ml-2" />
              محادثة جديدة
            </Button>
          </div>

          <div className="flex items-center gap-2 min-w-0">
            <Bot className="w-6 h-6 text-[#2E7D32] flex-shrink-0" />
            <div className="min-w-0">
              <div className="font-semibold text-gray-800 truncate">
                المساعد الذكي
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Messages */}
      <ScrollArea className="flex-1 min-h-0 p-4 lg:p-6 bg-white">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full space-y-6">
            <div className="w-24 h-24 bg-[#E8F5E9] rounded-full flex items-center justify-center">
              <Bot className="w-12 h-12 text-[#2E7D32]" />
            </div>

            <div className="text-center space-y-2 px-4">
              <h3 className="text-xl font-semibold text-gray-800">مرحباً بك!</h3>
              <p className="text-gray-600">كيف يمكنني مساعدتك اليوم؟</p>
            </div>
          </div>
        ) : (
          <div className="space-y-4 w-full px-2 lg:px-8">
            {messages.map((m) => (
              <div
                key={m.id}
                className={`flex ${
                  m.sender === "user" ? "justify-end" : "justify-start"
                }`}
              >
                <div
                  className={`w-fit max-w-[85%] sm:max-w-[70%] rounded-2xl p-4 break-words whitespace-pre-wrap ${
                    m.sender === "user"
                      ? "bg-[#2E7D32] text-white"
                      : "bg-gray-100 text-gray-800"
                  }`}
                >
                  {m.sender === "assistant" && (
                    <div className="flex items-center gap-2 mb-2">
                      <div className="w-6 h-6 bg-[#2E7D32] rounded-full flex items-center justify-center">
                        <Bot className="w-4 h-4 text-white" />
                      </div>
                      <span className="text-xs font-semibold text-[#2E7D32]">
                        المساعد الذكي
                      </span>
                    </div>
                  )}

                  {m.sender === "assistant" ? (
                    <div
                      className="
                        text-sm leading-relaxed break-words
                        [&_table]:w-full [&_table]:border-collapse
                        [&_th]:border [&_th]:border-gray-300 [&_th]:bg-white [&_th]:px-3 [&_th]:py-2 [&_th]:text-right
                        [&_td]:border [&_td]:border-gray-300 [&_td]:px-3 [&_td]:py-2 [&_td]:align-top [&_td]:text-right
                        [&_tr:nth-child(even)]:bg-white/60
                        [&_p]:m-0 [&_ul]:my-2 [&_ol]:my-2
                      "
                    >
                      <div className="overflow-x-auto">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {String(m.content || "")}
                        </ReactMarkdown>
                      </div>
                    </div>
                  ) : (
                    <p className="text-sm leading-relaxed break-words whitespace-pre-wrap">
                      {m.content}
                    </p>
                  )}

                  {/* ✅ خيارات اختيار المصدر كأزرار */}
                  {m.sender === "assistant" &&
                    Array.isArray(m.choices) &&
                    m.choices.length > 0 && (
                      <div className="mt-3 pt-3 border-t border-gray-200">
                        <p className="text-xs font-semibold text-gray-600 mb-2">
                         : اختر ملف 
                        </p>

                        <div className="flex flex-col gap-2">
                          {m.choices.map((c) => (
                            <Button
                              key={c.doc_key}
                              type="button"
                              variant="outline"
                              className="justify-start rounded-xl border-gray-300 bg-white hover:bg-gray-50 text-gray-800"
                              disabled={choosing || sending}
                              onClick={() => sendChoice(c.label, c.doc_key)}
                              title={c.label}
                            >
                              {choosing ? (
                                <span className="ml-2">⏳</span>
                              ) : (
                                <span className="ml-2">📄</span>
                              )}
                              <span className="truncate">{c.label}</span>
                            </Button>
                          ))}
                        </div>
                      </div>
                    )}

                  {/* Attachments */}
                  {m.sender === "assistant" &&
                    Array.isArray(m.attachments) &&
                    m.attachments.length > 0 && (
                      <div className="mt-3 pt-3 border-t border-gray-200 space-y-2">
                        <p className="text-xs font-semibold text-gray-600">
                          ملفات للتحميل
                        </p>

                        {m.attachments.map((a, idx) => {
                          const fullUrl = toAbsoluteUrl(a.url);

                          return (
                            <div
                              key={`${m.id}-att-${idx}`}
                              className="rounded-xl border border-gray-200 bg-white p-3"
                            >
                              <div className="flex items-start gap-3">
                                <div className="w-9 h-9 rounded-xl bg-[#E8F5E9] flex items-center justify-center flex-shrink-0">
                                  <FileText className="w-5 h-5 text-[#2E7D32]" />
                                </div>

                                <div className="min-w-0 flex-1">
                                  <div
                                    className="text-sm font-semibold text-gray-800 break-words"
                                    title={a.name}
                                  >
                                    {a.name}
                                  </div>

                                  {typeof a.size_mb === "number" && (
                                    <div className="text-xs text-gray-500 mt-1">
                                      {a.size_mb.toFixed(2)} MB
                                    </div>
                                  )}

                                  <a
                                    href={fullUrl}
                                    target="_blank"
                                    rel="noreferrer"
                                    className="inline-block mt-2"
                                  >
                                    <Button
                                      size="sm"
                                      variant="outline"
                                      className="rounded-xl border-[#2E7D32] text-[#2E7D32] hover:bg-[#E8F5E9]"
                                    >
                                      <Download className="w-4 h-4 ml-1" />
                                      تحميل
                                    </Button>
                                  </a>
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    )}

                  {/* Sources */}
                  {m.sender === "assistant" &&
                    Array.isArray(m.sources) &&
                    m.sources.length > 0 && (
                      <div className="mt-3 pt-3 border-t border-gray-200">
                        <p className="text-xs font-semibold text-gray-600 mb-2">
                          المصادر
                        </p>
                        <div className="flex flex-wrap gap-2">
                          {m.sources.map((s, idx) => (
                            <span
                              key={`${m.id}-src-${idx}`}
                              className="text-xs bg-white border border-gray-300 text-gray-700 rounded-full px-2 py-1"
                              title={s}
                            >
                              {s}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </ScrollArea>

      {/* Input */}
      <div className="border-t border-gray-200 p-3 lg:p-4 bg-white sticky bottom-0">
        <div className="max-w-4xl mx-auto flex gap-2">
          <Input
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendMessage(inputMessage)}
            placeholder="اكتب رسالتك هنا..."
            className="flex-1 rounded-xl border-gray-300 focus:border-[#2E7D32] focus:ring-[#2E7D32] h-11"
            disabled={sending || choosing}
          />
          <Button
            onClick={() => sendMessage(inputMessage)}
            className="bg-[#2E7D32] hover:bg-[#1B5E20] text-white rounded-xl px-6 h-11"
            disabled={sending || choosing}
            title={choosing ? "اختر ملف أولاً" : "إرسال"}
          >
            <Send className="h-5 w-5" />
          </Button>
        </div>
      </div>
    </div>
  );
}