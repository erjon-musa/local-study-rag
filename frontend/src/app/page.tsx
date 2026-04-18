"use client";

import { useRef, useState, useEffect, useCallback } from "react";
import Sidebar from "@/components/layout/Sidebar";
import ChatInput from "@/components/chat/ChatInput";
import ChatMessage from "@/components/chat/ChatMessage";
import SourceCard from "@/components/chat/SourceCard";
import CourseSelectorPill from "@/components/chat/CourseSelectorPill";
import {
  streamChat,
  type Source,
  type HistoryMessage,
} from "@/lib/api";
import {
  createCitationParser,
  parseCitations,
  type Segment,
} from "@/lib/citations";

/**
 * In-memory chat turn (per session — React state, not persisted).
 * `segments` is the rendered form; `content` is the raw text we echo back
 * to the backend as conversation history.
 */
interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  segments?: Segment[];
  sources?: Source[];
  isStreaming?: boolean;
  /** Distinguishes the empty-state "no content found" card from normal turns. */
  variant?: "default" | "empty-state";
}

const SUGGESTIONS = [
  "Explain the A* search algorithm",
  "What is a weak entity in ER diagrams?",
  "How does DDS content filtering work?",
  "Summarize the BFS vs DFS tradeoffs",
  "What are the key ML preprocessing steps?",
  "Explain CycloneDDS publish-subscribe",
];

/** Empty-state sentinel from backend/generation/prompts.py EMPTY_STATE_TEMPLATE. */
const EMPTY_STATE_PREFIX = "I don't see ";

/** How many turns of history to send back to the backend. Server also caps at 6. */
const HISTORY_WINDOW = 6;

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedCourse, setSelectedCourse] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [highlightedSource, setHighlightedSource] = useState<string | null>(
    null,
  );
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Abort + generation tracking. Tokens arriving with a stale generation
  // (e.g. after the user switched course mid-stream) are silently dropped.
  const abortRef = useRef<AbortController | null>(null);
  const generationRef = useRef(0);
  const toastTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const highlightTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Cleanup timers on unmount.
  useEffect(() => {
    return () => {
      if (toastTimerRef.current) clearTimeout(toastTimerRef.current);
      if (highlightTimerRef.current) clearTimeout(highlightTimerRef.current);
      abortRef.current?.abort();
    };
  }, []);

  const showToast = (text: string) => {
    setToast(text);
    if (toastTimerRef.current) clearTimeout(toastTimerRef.current);
    toastTimerRef.current = setTimeout(() => setToast(null), 3000);
  };

  const handleCitationClick = useCallback(
    (messageId: string, n: number) => {
      const domId = `source-card-${messageId}-${n}`;
      const el = document.getElementById(domId);
      if (!el) return;
      el.scrollIntoView({ behavior: "smooth", block: "center" });
      setHighlightedSource(domId);
      if (highlightTimerRef.current) clearTimeout(highlightTimerRef.current);
      highlightTimerRef.current = setTimeout(
        () => setHighlightedSource(null),
        1500,
      );
    },
    [],
  );

  const handleCourseChange = (newCourse: string | null) => {
    // Atomic switch: abort any in-flight stream, bump generation so late
    // tokens from the prior fetch are ignored, clear messages, toast.
    abortRef.current?.abort();
    abortRef.current = null;
    generationRef.current += 1;
    setMessages([]);
    setIsLoading(false);
    setSelectedCourse(newCourse);
    const label = newCourse ?? "All Courses";
    showToast(`Switched to ${label} — starting a fresh conversation`);
  };

  const handleSend = async (question: string) => {
    if (isLoading) return;

    // Take a snapshot of the send-time generation and abort any prior request.
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    generationRef.current += 1;
    const myGeneration = generationRef.current;

    const userId = `user-${Date.now()}`;
    const assistantId = `assistant-${Date.now()}`;

    const userMessage: Message = {
      id: userId,
      role: "user",
      content: question,
    };

    const assistantMessage: Message = {
      id: assistantId,
      role: "assistant",
      content: "",
      segments: [],
      sources: [],
      isStreaming: true,
    };

    // Build the history payload from the state BEFORE appending this turn.
    // Only content + sources are needed (the backend strips citations for
    // history messages anyway via _neutralize_citations).
    const historyPayload: HistoryMessage[] = messages
      .slice(-HISTORY_WINDOW)
      .map((m) => ({
        role: m.role,
        content: m.content,
        ...(m.sources && m.sources.length > 0 ? { sources: m.sources } : {}),
      }));

    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setIsLoading(true);

    const parser = createCitationParser();
    let rawContent = "";
    let emptyStateDecided = false;
    let isEmptyState = false;

    try {
      const stream = streamChat(
        question,
        selectedCourse || undefined,
        10,
        historyPayload,
        controller.signal,
      );

      for await (const event of stream) {
        // Drop any event that belongs to a superseded request.
        if (myGeneration !== generationRef.current) continue;

        if (event.type === "sources") {
          const srcs = (event.data as Source[]) ?? [];
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, sources: srcs } : m,
            ),
          );
        } else if (event.type === "token") {
          const token = (event.data as string) ?? "";
          rawContent += token;

          // Detect the empty-state sentinel on the first meaningful tokens.
          // We wait until we have enough characters to match the prefix.
          if (!emptyStateDecided && rawContent.length >= EMPTY_STATE_PREFIX.length) {
            isEmptyState = rawContent.startsWith(EMPTY_STATE_PREFIX);
            emptyStateDecided = true;
          }

          const newSegments = parser.push(token);
          if (newSegments.length > 0) {
            setMessages((prev) =>
              prev.map((m) => {
                if (m.id !== assistantId) return m;
                const merged = mergeSegments(m.segments ?? [], newSegments);
                return {
                  ...m,
                  content: rawContent,
                  segments: merged,
                  variant: isEmptyState ? "empty-state" : "default",
                };
              }),
            );
          }
        } else if (event.type === "done") {
          const tail = parser.flush();
          setMessages((prev) =>
            prev.map((m) => {
              if (m.id !== assistantId) return m;
              const merged =
                tail.length > 0
                  ? mergeSegments(m.segments ?? [], tail)
                  : m.segments;
              return {
                ...m,
                content: rawContent,
                segments: merged,
                isStreaming: false,
                variant: isEmptyState ? "empty-state" : "default",
              };
            }),
          );
        }
      }
    } catch (error) {
      // Ignore aborts — a fresh request (or course switch) is already in flight.
      if ((error as Error)?.name === "AbortError") {
        return;
      }
      if (myGeneration === generationRef.current) {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? {
                  ...m,
                  content:
                    "⚠️ Something went wrong. Make sure the backend is running and LM Studio is active.",
                  segments: parseCitations(
                    "⚠️ Something went wrong. Make sure the backend is running and LM Studio is active.",
                  ),
                  isStreaming: false,
                }
              : m,
          ),
        );
      }
    } finally {
      if (myGeneration === generationRef.current) {
        setIsLoading(false);
        abortRef.current = null;
      }
    }
  };

  return (
    <div className="app-layout">
      <Sidebar
        selectedCourse={selectedCourse}
        onCourseSelect={handleCourseChange}
      />

      <div className="main-content">
        <div className="chat-container">
          <div className="chat-header">
            <CourseSelectorPill
              selectedCourse={selectedCourse}
              onChange={handleCourseChange}
            />
            <h1 className="chat-title">Study Assistant</h1>
            <p className="chat-subtitle">
              Ask questions about your course notes · Powered by Gemma4
            </p>
          </div>

          {toast && (
            <div className="chat-toast" role="status" aria-live="polite">
              {toast}
            </div>
          )}

          <div className="messages-container" id="messages-container">
            {messages.length === 0 ? (
              <div className="empty-state">
                <div className="empty-state-icon">🎓</div>
                <h2 className="empty-state-title">Ready to study?</h2>
                <p className="empty-state-subtitle">
                  Ask me anything about your course materials. I&apos;ll find the
                  relevant sections and give you cited answers.
                </p>
                <div className="suggestion-chips">
                  {SUGGESTIONS.map((suggestion) => (
                    <button
                      key={suggestion}
                      className="suggestion-chip"
                      onClick={() => handleSend(suggestion)}
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              messages.map((msg) => {
                const isEmpty = msg.variant === "empty-state";
                return (
                  <div key={msg.id}>
                    <ChatMessage
                      role={msg.role}
                      content={msg.role === "user" ? msg.content : undefined}
                      segments={msg.role === "assistant" ? msg.segments : undefined}
                      isStreaming={msg.isStreaming}
                      variant={msg.variant}
                      onCitationClick={(n) => handleCitationClick(msg.id, n)}
                    />
                    {msg.role === "assistant" &&
                      msg.sources &&
                      msg.sources.length > 0 && (
                        <div className="sources-container">
                          {isEmpty && (
                            <div className="sources-label">Closest matches</div>
                          )}
                          <div className="sources-row">
                            {msg.sources.map((source, i) => {
                              const domId = `source-card-${msg.id}-${i + 1}`;
                              return (
                                <SourceCard
                                  key={domId}
                                  source={source}
                                  index={i + 1}
                                  domId={domId}
                                  highlighted={highlightedSource === domId}
                                />
                              );
                            })}
                          </div>
                        </div>
                      )}
                  </div>
                );
              })
            )}

            {isLoading &&
              messages[messages.length - 1]?.content === "" && (
                <div className="message message-assistant">
                  <span className="message-label">Study Assistant</span>
                  <div className="message-bubble">
                    <div className="typing-indicator">
                      <span className="typing-dot" />
                      <span className="typing-dot" />
                      <span className="typing-dot" />
                    </div>
                  </div>
                </div>
              )}

            <div ref={messagesEndRef} />
          </div>

          <ChatInput
            onSend={handleSend}
            disabled={isLoading}
            courseName={selectedCourse}
          />
        </div>
      </div>
    </div>
  );
}

/**
 * Append `next` segments to `prev`, merging adjacent text segments so React
 * doesn't have to reconcile a long string of one-char spans during streaming.
 */
function mergeSegments(prev: Segment[], next: Segment[]): Segment[] {
  if (next.length === 0) return prev;
  const out = prev.slice();
  for (const seg of next) {
    const last = out[out.length - 1];
    if (seg.type === "text" && last && last.type === "text") {
      out[out.length - 1] = { type: "text", value: last.value + seg.value };
    } else {
      out.push({ ...seg });
    }
  }
  return out;
}
