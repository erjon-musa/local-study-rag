"use client";

import { useState, type KeyboardEvent } from "react";

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  courseName?: string | null;
}

export default function ChatInput({ onSend, disabled, courseName }: ChatInputProps) {
  const [message, setMessage] = useState("");

  const handleSend = () => {
    const trimmed = message.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setMessage("");
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="chat-input-container">
      <div className="chat-input-wrapper">
        {courseName && (
          <span className="course-filter">
            🎯 {courseName.split(" - ")[0]}
          </span>
        )}
        <textarea
          className="chat-input"
          placeholder="Ask about your notes…"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          rows={1}
          id="chat-message-input"
        />
        <button
          className="chat-send-btn"
          onClick={handleSend}
          disabled={disabled || !message.trim()}
          id="chat-send-button"
          aria-label="Send message"
        >
          ↑
        </button>
      </div>
    </div>
  );
}
