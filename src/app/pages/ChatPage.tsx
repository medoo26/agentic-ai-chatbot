import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { ScrollArea } from "../components/ui/scroll-area";
import { Dialog, DialogContent } from "../components/ui/dialog";
import { ArrowUp, LogOut, Plus, Bot, Download, FileText, Eye } from "lucide-react";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// API
import { apiPostChatStream, API_BASE_URL, type ChatStreamMeta } from "../../api";

type Attachment = {
  name: string;
  url: string;
  mime?: string;
  size_mb?: number;
};

type ChoiceItem = {
  label: string;
  doc_key: string;
};

type SourceItem = {
  name: string;
  page?: number | null;
  public_id?: string | null;
  url?: string | null;
  mime?: string | null;
};

type UIMessage = {
  id: string;
  content: string;
  sender: "user" | "assistant";
  timestamp: Date;
  sources?: SourceItem[];
  attachments?: Attachment[];
  choices?: ChoiceItem[];
  debug?: string;
};

export function ChatPage() {
  const navigate = useNavigate();

  const [messages, setMessages] = useState<UIMessage[]>([]);
  const [inputMessage, setInputMessage] = useState("");
  const [sending, setSending] = useState(false);
  const [choosing, setChoosing] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [previewOpen, setPreviewOpen] = useState(false);
  const [previewSource, setPreviewSource] = useState<SourceItem | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const activeStreamRef = useRef<AbortController | null>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    const id = window.requestAnimationFrame(() => {
      scrollToBottom();
    });
    return () => window.cancelAnimationFrame(id);
  }, [messages]);

  const handleLogout = () => navigate("/");

  const createNewChat = () => {
    activeStreamRef.current?.abort();
    activeStreamRef.current = null;
    setMessages([]);
    setInputMessage("");
    setStreaming(false);
  };

  const toAbsoluteUrl = (url: string) => {
    if (!url) return "";
    if (/^https?:\/\//i.test(url)) return url;
    return `${API_BASE_URL}${url.startsWith("/") ? url : `/${url}`}`;
  };

  const cleanAssistantContent = (content: string) => {
    const text = String(content || "");
    // Hide internal source-title marker if the model emits it.
    return text
      .replace(/\n?\s*SOURCES_TITLES_JSON:\s*\[[^\]]*\]\s*$/gim, "")
      .replace(/\n{3,}/g, "\n\n")
      .trimEnd();
  };

  const formatSourceName = (name: string) => {
    const value = String(name || "").trim();
    return value.replace(/\.(pdf|docx|doc|txt|xlsx|xls|pptx|ppt)$/i, "");
  };

  const getSourcePreviewUrl = (source: SourceItem) => {
    const rawUrl = String(source?.url || "").trim();
    if (rawUrl) {
      const absolute = toAbsoluteUrl(rawUrl);
      if (source?.page && Number(source.page) > 0) {
        return `${absolute}#page=${source.page}`;
      }
      return absolute;
    }
    const pid = String(source?.public_id || "").trim();
    if (!pid) return "";
    const absolute = toAbsoluteUrl(`/api/files/${pid}/preview`);
    if (source?.page && Number(source.page) > 0) {
      return `${absolute}#page=${source.page}`;
    }
    return absolute;
  };

  const openSourcePreview = (source: SourceItem) => {
    const mime = String(source?.mime || "").toLowerCase();
    const isPdf = !mime || mime.includes("pdf");
    const previewUrl = getSourcePreviewUrl(source);
    if (!isPdf || !previewUrl) return;
    setPreviewSource(source);
    setPreviewOpen(true);
  };

  const sendMessage = async (content: string) => {
    if (!content.trim() || sending || choosing || streaming) return;
    setSending(true);
    setStreaming(true);

    const userMsg: UIMessage = {
      id: `u-${Date.now()}`,
      content,
      sender: "user",
      timestamp: new Date(),
      sources: [],
      attachments: [],
      choices: [],
      debug: "",
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputMessage("");
    const assistantId = `a-stream-${Date.now()}`;
    setMessages((prev) => [
      ...prev,
      {
        id: assistantId,
        content: "",
        sender: "assistant",
        timestamp: new Date(),
        sources: [],
        attachments: [],
        choices: [],
        debug: "",
      },
    ]);

    const controller = new AbortController();
    activeStreamRef.current = controller;

    try {
      await apiPostChatStream("/api/chat/stream", { content }, {
        signal: controller.signal,
        onToken: (delta) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, content: `${m.content || ""}${delta}` } : m
            )
          );
        },
        onMeta: (meta: ChatStreamMeta) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? {
                    ...m,
                    timestamp: meta?.timestamp ? new Date(meta.timestamp) : m.timestamp,
                    sources: Array.isArray(meta?.sources) ? meta.sources : [],
                    attachments: Array.isArray(meta?.attachments) ? meta.attachments : [],
                    choices: Array.isArray(meta?.choices) ? meta.choices : [],
                    debug: meta?.debug || "",
                  }
                : m
            )
          );
        },
      });
    } catch (err: any) {
      const message = `صار خطأ: ${err?.message || "غير معروف"}`;
      setMessages((prev) =>
        prev.map((m) => (m.id === assistantId ? { ...m, content: m.content || message } : m))
      );
    } finally {
      activeStreamRef.current = null;
      setStreaming(false);
      setSending(false);
    }
  };

  const sendChoice = async (label: string, doc_key: string) => {
    if (!label || !doc_key || sending || choosing || streaming) return;

    setChoosing(true);
    setStreaming(true);

    const userMsg: UIMessage = {
      id: `u-choice-${Date.now()}`,
      content: label,
      sender: "user",
      timestamp: new Date(),
      sources: [],
      attachments: [],
      choices: [],
      debug: "",
    };
    setMessages((prev) => [...prev, userMsg]);
    const assistantId = `a-choice-stream-${Date.now()}`;
    setMessages((prev) => [
      ...prev,
      {
        id: assistantId,
        content: "",
        sender: "assistant",
        timestamp: new Date(),
        sources: [],
        attachments: [],
        choices: [],
        debug: "",
      },
    ]);
    const controller = new AbortController();
    activeStreamRef.current = controller;

    try {
      await apiPostChatStream(
        "/api/chat/stream",
        { content: label, choice_doc_key: doc_key },
        {
          signal: controller.signal,
          onToken: (delta) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId ? { ...m, content: `${m.content || ""}${delta}` } : m
              )
            );
          },
          onMeta: (meta: ChatStreamMeta) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? {
                      ...m,
                      timestamp: meta?.timestamp ? new Date(meta.timestamp) : m.timestamp,
                      sources: Array.isArray(meta?.sources) ? meta.sources : [],
                      attachments: Array.isArray(meta?.attachments) ? meta.attachments : [],
                      choices: Array.isArray(meta?.choices) ? meta.choices : [],
                      debug: meta?.debug || "",
                    }
                  : m
              )
            );
          },
        }
      );
    } catch (err: any) {
      const message = `صار خطأ: ${err?.message || "غير معروف"}`;
      setMessages((prev) =>
        prev.map((m) => (m.id === assistantId ? { ...m, content: m.content || message } : m))
      );
    } finally {
      activeStreamRef.current = null;
      setStreaming(false);
      setChoosing(false);
    }
  };

  return (
    <>
    <div className="h-dvh flex flex-col bg-white overflow-hidden" dir="rtl">
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
                className="flex justify-end"
              >
                <div
                  className={`w-fit max-w-[85%] sm:max-w-[70%] rounded-2xl p-4 break-words whitespace-pre-wrap text-right ${
                    m.sender === "assistant"
                      ? "bg-transparent text-black p-0 rounded-none"
                      : "bg-gray-100 text-gray-800 border border-gray-200"
                  }`}
                  dir="rtl"
                >
                  {m.sender === "assistant" ? (
                    <div
                      dir="rtl"
                      className="
                        text-sm leading-relaxed break-words text-right text-black
                        [&_*]:text-right
                        [&_table]:w-full [&_table]:border-collapse
                        [&_th]:border [&_th]:border-gray-300 [&_th]:bg-gray-100 [&_th]:px-3 [&_th]:py-2 [&_th]:text-right
                        [&_td]:border [&_td]:border-gray-300 [&_td]:px-3 [&_td]:py-2 [&_td]:align-top [&_td]:text-right
                        [&_tr:nth-child(even)]:bg-gray-50
                        [&_p]:m-0 [&_p]:text-right
                        [&_ul]:my-2 [&_ul]:pr-5 [&_ul]:text-right
                        [&_ol]:my-2 [&_ol]:pr-5 [&_ol]:text-right
                        [&_li]:text-right
                        [&_a]:text-gray-900 [&_a]:underline
                        [&_strong]:text-black
                        [&_code]:bg-gray-100 [&_code]:px-1 [&_code]:py-0.5 [&_code]:rounded
                      "
                    >
                      <div className="overflow-x-auto">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {cleanAssistantContent(String(m.content || ""))}
                        </ReactMarkdown>
                      </div>
                    </div>
                  ) : (
                    <p
                      dir="rtl"
                      className="text-sm leading-relaxed break-words whitespace-pre-wrap text-right"
                    >
                      {m.content}
                    </p>
                  )}

                  {m.sender === "assistant" &&
                    Array.isArray(m.choices) &&
                    m.choices.length > 0 && (
                      <div className="mt-3 pt-3 border-t border-gray-200">
                        <p className="text-xs font-semibold text-gray-700 mb-2">
                          اختر ملف:
                        </p>

                        <div className="flex flex-col gap-2">
                          {m.choices.map((c) => (
                            <Button
                              key={c.doc_key}
                              type="button"
                              variant="outline"
                              className="justify-start rounded-xl border-white/30 bg-white text-gray-800 hover:bg-gray-100"
                              disabled={choosing || sending || streaming}
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

                  {m.sender === "assistant" &&
                    Array.isArray(m.attachments) &&
                    m.attachments.length > 0 && (
                      <div className="mt-3 pt-3 border-t border-gray-200 space-y-2">
                        <p className="text-xs font-semibold text-gray-700">
                          ملفات للتحميل
                        </p>

                        {m.attachments.map((a, idx) => {
                          const fullUrl = toAbsoluteUrl(a.url);

                          return (
                            <div
                              key={`${m.id}-att-${idx}`}
                              className="rounded-xl border border-white/20 bg-white p-3"
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

                  {m.sender === "assistant" &&
                    Array.isArray(m.sources) &&
                    m.sources.length > 0 && (
                      <div className="mt-3 pt-3 border-t border-gray-200">
                        <p className="text-xs font-semibold text-gray-700 mb-2">
                          المصادر
                        </p>
                        <div className="flex flex-wrap gap-2">
                          {m.sources.map((s, idx) => (
                            <button
                              key={`${m.id}-src-${idx}`}
                              type="button"
                              className="inline-flex items-center gap-1 text-xs bg-white text-gray-700 border border-gray-300 rounded-full px-2 py-1 hover:bg-gray-50 transition-colors"
                              title={s.page ? `${s.name} — صفحة ${s.page}` : s.name}
                              onClick={() => openSourcePreview(s)}
                            >
                              <Eye className="w-3 h-3" />
                              <span>معاينة</span>
                              <span className="text-gray-400">|</span>
                              {formatSourceName(s.name)}
                              {s.page ? ` — صفحة ${s.page}` : ""}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}

                  {m.sender === "assistant" && m.debug && (
                    <div className="mt-3 pt-3 border-t border-gray-200">
                      <p className="text-xs font-semibold text-gray-700 mb-2">
                        Debug (RAG)
                      </p>
                      <pre className="text-xs bg-black/30 text-white p-3 rounded-lg overflow-x-auto whitespace-pre-wrap text-left dir-ltr">
                        {m.debug}
                      </pre>
                    </div>
                  )}
                </div>
              </div>
            ))}
            {(sending || choosing || streaming) && (
              <div className="flex justify-end">
                <div className="inline-flex items-center gap-1 rounded-full bg-gray-100 border border-gray-200 px-3 py-2">
                  <span
                    className="w-2 h-2 rounded-full bg-gray-500 animate-bounce"
                    style={{ animationDelay: "0ms" }}
                  />
                  <span
                    className="w-2 h-2 rounded-full bg-gray-500 animate-bounce"
                    style={{ animationDelay: "150ms" }}
                  />
                  <span
                    className="w-2 h-2 rounded-full bg-gray-500 animate-bounce"
                    style={{ animationDelay: "300ms" }}
                  />
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </ScrollArea>

      <div className="border-t border-gray-200 p-3 lg:p-4 bg-white sticky bottom-0">
        <div className="max-w-4xl mx-auto flex gap-2">
          <Input
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendMessage(inputMessage)}
            placeholder="اكتب رسالتك هنا..."
            className="flex-1 rounded-xl border-gray-300 focus:border-[#2E7D32] focus:ring-[#2E7D32] h-11"
            disabled={sending || choosing || streaming}
          />
          <Button
            onClick={() => sendMessage(inputMessage)}
            className="bg-[#2E7D32] hover:bg-[#1B5E20] text-white rounded-full w-11 h-11 p-0 flex items-center justify-center"
            disabled={sending || choosing || streaming}
            title={choosing ? "اختر ملف أولاً" : "إرسال"}
          >
            <ArrowUp className="h-5 w-5" />
          </Button>
        </div>
      </div>
    </div>
    <Dialog open={previewOpen} onOpenChange={setPreviewOpen}>
      <DialogContent className="w-[100vw] h-[100dvh] max-w-none rounded-none p-2 gap-2 sm:w-[98vw] sm:h-[94vh] sm:max-w-7xl sm:rounded-lg sm:p-4">
        <div className="flex-1 min-h-0 w-full">
          {previewSource ? (
            <iframe
              src={getSourcePreviewUrl(previewSource)}
              title={formatSourceName(previewSource.name)}
              className="w-full h-full border rounded-md sm:rounded-lg"
            />
          ) : null}
        </div>
      </DialogContent>
    </Dialog>
    </>
  );
}