"use client";

import { type Segment } from "@/lib/citations";

interface ChatMessageProps {
  role: "user" | "assistant";
  /** Pre-parsed segments (text + citation chips). Preferred path. */
  segments?: Segment[];
  /** Fallback raw content (used for user messages and error states). */
  content?: string;
  isStreaming?: boolean;
  /** Called when the user clicks a citation chip — parent scrolls+highlights. */
  onCitationClick?: (n: number) => void;
  /** Render as a distinct empty-state card instead of the normal bubble. */
  variant?: "default" | "empty-state";
}

export default function ChatMessage({
  role,
  segments,
  content,
  isStreaming,
  onCitationClick,
  variant = "default",
}: ChatMessageProps) {
  const isEmptyState = variant === "empty-state" && role === "assistant";

  return (
    <div className={`message message-${role}`}>
      <span className="message-label">
        {role === "user" ? "You" : "Study Assistant"}
      </span>
      <div
        className={
          isEmptyState
            ? "message-bubble empty-state-card"
            : "message-bubble"
        }
      >
        {isEmptyState && (
          <span className="empty-state-card-icon" aria-hidden="true">
            🔍
          </span>
        )}
        <span className="message-body">
          {segments
            ? segments.map((seg, i) =>
                seg.type === "text" ? (
                  <span key={i}>{seg.value}</span>
                ) : (
                  <sup
                    key={i}
                    className="citation-chip"
                    role="button"
                    tabIndex={0}
                    onClick={() => onCitationClick?.(seg.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        onCitationClick?.(seg.value);
                      }
                    }}
                    aria-label={`Jump to source ${seg.value}`}
                  >
                    [{seg.value}]
                  </sup>
                ),
              )
            : content}
          {isStreaming && (
            <span style={{ opacity: 0.5, animation: "blink 1s infinite" }}>▋</span>
          )}
        </span>
      </div>
    </div>
  );
}
