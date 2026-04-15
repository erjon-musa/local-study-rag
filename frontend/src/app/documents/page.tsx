"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Sidebar from "@/components/layout/Sidebar";
import {
  getStats,
  listCourses,
  listDocuments,
  syncVault,
  uploadFile,
  type CourseInfo,
  type DocumentInfo,
  type StatsResult,
  type SyncResult,
} from "@/lib/api";

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [courses, setCourses] = useState<CourseInfo[]>([]);
  const [stats, setStats] = useState<StatsResult | null>(null);
  const [selectedCourse, setSelectedCourse] = useState<string | null>(null);
  const [syncing, setSyncing] = useState(false);
  const [syncStatus, setSyncStatus] = useState<{
    type: "success" | "error" | "loading";
    message: string;
  } | null>(null);
  const [dragging, setDragging] = useState(false);
  const [uploadCourse, setUploadCourse] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadData = useCallback(async () => {
    try {
      const [docsData, coursesData, statsData] = await Promise.all([
        listDocuments(selectedCourse || undefined),
        listCourses(),
        getStats(),
      ]);
      setDocuments(docsData.documents);
      setCourses(coursesData.courses);
      setStats(statsData);
    } catch {
      // Backend might not be running
    }
  }, [selectedCourse]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleSync = async () => {
    setSyncing(true);
    setSyncStatus({ type: "loading", message: "Scanning vault for changes…" });

    try {
      const result: SyncResult = await syncVault();
      const parts = [];
      if (result.new > 0) parts.push(`${result.new} new`);
      if (result.updated > 0) parts.push(`${result.updated} updated`);
      if (result.deleted > 0) parts.push(`${result.deleted} deleted`);
      if (parts.length === 0) parts.push("Everything up to date");

      setSyncStatus({
        type: result.errors.length > 0 ? "error" : "success",
        message: `${parts.join(", ")} · ${result.duration_seconds}s`,
      });
      await loadData();
    } catch (err) {
      setSyncStatus({
        type: "error",
        message: "Failed to sync. Is the backend running?",
      });
    } finally {
      setSyncing(false);
      setTimeout(() => setSyncStatus(null), 5000);
    }
  };

  const handleFileDrop = async (files: FileList) => {
    if (!uploadCourse) {
      setSyncStatus({
        type: "error",
        message: "Select a course first before uploading",
      });
      setTimeout(() => setSyncStatus(null), 3000);
      return;
    }

    setSyncStatus({
      type: "loading",
      message: `Uploading ${files.length} file(s)…`,
    });

    try {
      for (const file of Array.from(files)) {
        await uploadFile(file, uploadCourse, "Resources");
      }
      setSyncStatus({
        type: "success",
        message: `Uploaded ${files.length} file(s) to ${uploadCourse.split(" - ")[0]}`,
      });
      await loadData();
    } catch {
      setSyncStatus({ type: "error", message: "Upload failed" });
    }
    setTimeout(() => setSyncStatus(null), 5000);
  };

  const courseColors: Record<string, string> = {
    "CMPE 223 - Software Specification": "var(--course-cmpe223)",
    "ELEC 472 - Artificial Intelligence": "var(--course-elec472)",
    "ELEC 477 - Distributed Systems": "var(--course-elec477)",
  };

  return (
    <div className="app-layout">
      <Sidebar
        selectedCourse={selectedCourse}
        onCourseSelect={setSelectedCourse}
      />

      <div className="main-content">
        <div className="docs-container">
          <div className="docs-header">
            <h1 className="docs-title">Documents</h1>
            <div className="docs-actions">
              <button
                className="btn btn-secondary"
                onClick={handleSync}
                disabled={syncing}
                id="sync-button"
              >
                {syncing ? (
                  <>
                    <span className="spinner" /> Syncing…
                  </>
                ) : (
                  <>🔄 Sync Vault</>
                )}
              </button>
            </div>
          </div>

          {syncStatus && (
            <div className={`sync-status ${syncStatus.type}`}>
              {syncStatus.type === "loading" && <span className="spinner" />}
              {syncStatus.type === "success" && "✓"}
              {syncStatus.type === "error" && "✗"}
              {syncStatus.message}
            </div>
          )}

          {/* Stats */}
          {stats && (
            <div className="stats-grid">
              <div className="stat-card">
                <div className="stat-value">{stats.total_files}</div>
                <div className="stat-label">Documents</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">{stats.total_chunks}</div>
                <div className="stat-label">Chunks</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">{Object.keys(stats.courses).length}</div>
                <div className="stat-label">Courses</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">{stats.collection_count}</div>
                <div className="stat-label">Vectors</div>
              </div>
            </div>
          )}

          {/* Upload Zone */}
          <div
            className={`upload-zone ${dragging ? "dragging" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              if (e.dataTransfer.files.length > 0) {
                handleFileDrop(e.dataTransfer.files);
              }
            }}
            onClick={() => fileInputRef.current?.click()}
          >
            <div className="upload-icon">📁</div>
            <div className="upload-text">
              Drag &amp; drop files here or click to browse
            </div>
            <div className="upload-hint">
              PDF, DOCX, TXT, MD, HTML supported
            </div>
            <div style={{ marginTop: "12px", display: "flex", gap: "8px", justifyContent: "center", flexWrap: "wrap" }}>
              <select
                value={uploadCourse}
                onChange={(e) => setUploadCourse(e.target.value)}
                onClick={(e) => e.stopPropagation()}
                style={{
                  background: "var(--bg-tertiary)",
                  border: "1px solid var(--border)",
                  borderRadius: "6px",
                  padding: "6px 10px",
                  color: "var(--text-secondary)",
                  fontSize: "12px",
                  fontFamily: "inherit",
                }}
                id="upload-course-select"
              >
                <option value="">Select course…</option>
                {courses.map((c) => (
                  <option key={c.name} value={c.name}>
                    {c.name}
                  </option>
                ))}
              </select>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".pdf,.docx,.txt,.md,.html"
              style={{ display: "none" }}
              onChange={(e) => {
                if (e.target.files && e.target.files.length > 0) {
                  handleFileDrop(e.target.files);
                }
              }}
            />
          </div>

          {/* Document Table */}
          {documents.length > 0 ? (
            <table className="docs-table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Course</th>
                  <th>Category</th>
                  <th>Chunks</th>
                  <th>Indexed</th>
                </tr>
              </thead>
              <tbody>
                {documents.map((doc) => (
                  <tr key={doc.path}>
                    <td className="doc-name">{doc.name}</td>
                    <td>
                      <span
                        className="course-tag"
                        style={{
                          background: `${courseColors[doc.course] || "var(--accent-blue)"}20`,
                          color: courseColors[doc.course] || "var(--accent-blue)",
                        }}
                      >
                        {doc.course.split(" - ")[0]}
                      </span>
                    </td>
                    <td>{doc.category}</td>
                    <td>{doc.chunks}</td>
                    <td style={{ fontSize: "11px", color: "var(--text-muted)" }}>
                      {doc.ingested_at
                        ? new Date(doc.ingested_at).toLocaleDateString()
                        : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="empty-state" style={{ padding: "48px 0" }}>
              <div className="empty-state-icon">📭</div>
              <h2 className="empty-state-title">No documents indexed</h2>
              <p className="empty-state-subtitle">
                Click &quot;Sync Vault&quot; to ingest your study materials, or drag
                files into the upload zone above.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
