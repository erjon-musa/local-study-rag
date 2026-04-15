"use client";

import { useState } from "react";
import type { Source } from "@/lib/api";

interface SourceCardProps {
  source: Source;
}

export default function SourceCard({ source }: SourceCardProps) {
  const [expanded, setExpanded] = useState(false);

  const fileIcon = () => {
    const ext = source.filename.split(".").pop()?.toLowerCase();
    switch (ext) {
      case "pdf": return "📕";
      case "txt": return "📝";
      case "md": return "📘";
      case "docx": return "📄";
      case "html": return "🌐";
      default: return "📎";
    }
  };

  const scorePercent = Math.round(source.relevance_score * 10000) / 100;

  return (
    <>
      <div className="source-card" onClick={() => setExpanded(true)}>
        <span className="source-icon">{fileIcon()}</span>
        <div className="source-info">
          <div className="source-name">{source.filename}</div>
          <div className="source-meta">
            {source.page ? `Page ${source.page}` : source.category}
            {source.course && ` · ${source.course.split(" - ")[0]}`}
          </div>
        </div>
        <span className="source-score">{scorePercent.toFixed(1)}%</span>
      </div>

      {expanded && (
        <div className="source-expanded" onClick={() => setExpanded(false)}>
          <div
            className="source-expanded-content"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="source-expanded-header">
              <div>
                <div className="source-expanded-title">
                  {fileIcon()} {source.filename}
                </div>
                <div style={{ fontSize: "12px", color: "var(--text-muted)", marginTop: "4px" }}>
                  {source.course} · {source.category}
                  {source.page && ` · Page ${source.page}`}
                  {" · "}Relevance: {scorePercent.toFixed(1)}%
                </div>
              </div>
              <button
                className="source-expanded-close"
                onClick={() => setExpanded(false)}
              >
                ✕
              </button>
            </div>
            <div className="source-expanded-preview">
              {source.text_preview}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
