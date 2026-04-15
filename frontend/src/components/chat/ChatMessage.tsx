"use client";

interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
  isStreaming?: boolean;
}

export default function ChatMessage({ role, content, isStreaming }: ChatMessageProps) {
  return (
    <div className={`message message-${role}`}>
      <span className="message-label">
        {role === "user" ? "You" : "Study Assistant"}
      </span>
      <div className="message-bubble">
        {content}
        {isStreaming && (
          <span style={{ opacity: 0.5, animation: "blink 1s infinite" }}>▋</span>
        )}
      </div>
    </div>
  );
}
