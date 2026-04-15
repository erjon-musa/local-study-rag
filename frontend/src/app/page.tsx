"use client";

import { useRef, useState, useEffect } from "react";
import Sidebar from "@/components/layout/Sidebar";
import ChatInput from "@/components/chat/ChatInput";
import ChatMessage from "@/components/chat/ChatMessage";
import SourceCard from "@/components/chat/SourceCard";
import { streamChat, type Source } from "@/lib/api";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  isStreaming?: boolean;
}

const SUGGESTIONS = [
  "Explain the A* search algorithm",
  "What is a weak entity in ER diagrams?",
  "How does DDS content filtering work?",
  "Summarize the BFS vs DFS tradeoffs",
  "What are the key ML preprocessing steps?",
  "Explain CycloneDDS publish-subscribe",
];

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedCourse, setSelectedCourse] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async (question: string) => {
    if (isLoading) return;

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: "user",
      content: question,
    };

    const assistantMessage: Message = {
      id: `assistant-${Date.now()}`,
      role: "assistant",
      content: "",
      sources: [],
      isStreaming: true,
    };

    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setIsLoading(true);

    try {
      const stream = streamChat(question, selectedCourse || undefined);

      for await (const event of stream) {
        if (event.type === "sources") {
          setMessages((prev) => {
            const updated = [...prev];
            const lastIndex = updated.length - 1;
            const last = updated[lastIndex];
            if (last.role === "assistant") {
              updated[lastIndex] = { ...last, sources: event.data as Source[] };
            }
            return updated;
          });
        } else if (event.type === "token") {
          setMessages((prev) => {
            const updated = [...prev];
            const lastIndex = updated.length - 1;
            const last = updated[lastIndex];
            if (last.role === "assistant") {
              updated[lastIndex] = { ...last, content: last.content + (event.data as string) };
            }
            return updated;
          });
        } else if (event.type === "done") {
          setMessages((prev) => {
            const updated = [...prev];
            const lastIndex = updated.length - 1;
            const last = updated[lastIndex];
            if (last.role === "assistant") {
              updated[lastIndex] = { ...last, isStreaming: false };
            }
            return updated;
          });
        }
      }
    } catch (error) {
      setMessages((prev) => {
        const updated = [...prev];
        const lastIndex = updated.length - 1;
        const last = updated[lastIndex];
        if (last.role === "assistant") {
          updated[lastIndex] = {
            ...last,
            content: "⚠️ Something went wrong. Make sure the backend is running and LM Studio is active.",
            isStreaming: false
          };
        }
        return updated;
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-layout">
      <Sidebar
        selectedCourse={selectedCourse}
        onCourseSelect={setSelectedCourse}
      />

      <div className="main-content">
        <div className="chat-container">
          <div className="chat-header">
            <h1 className="chat-title">Study Assistant</h1>
            <p className="chat-subtitle">
              Ask questions about your course notes · Powered by Gemma4
            </p>
          </div>

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
              messages.map((msg) => (
                <div key={msg.id}>
                  <ChatMessage
                    role={msg.role}
                    content={msg.content}
                    isStreaming={msg.isStreaming}
                  />
                  {msg.role === "assistant" &&
                    msg.sources &&
                    msg.sources.length > 0 && (
                      <div className="sources-container">
                        {msg.sources.map((source, i) => (
                          <SourceCard key={`${source.filename}-${i}`} source={source} />
                        ))}
                      </div>
                    )}
                </div>
              ))
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
