/**
 * API client for the RAG backend.
 * All requests go to the FastAPI server at localhost:8000.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ── Types ────────────────────────────────────────────────────

export interface Source {
  filename: string;
  page: string;
  course: string;
  category: string;
  relevance_score: number;
  text_preview: string;
}

export interface ChatEvent {
  type: "sources" | "token" | "done";
  data?: Source[] | string;
}

export interface DocumentInfo {
  name: string;
  path: string;
  course: string;
  category: string;
  chunks: number;
  ingested_at: string;
}

export interface CourseInfo {
  name: string;
  files: number;
  chunks: number;
}

export interface SyncResult {
  new: number;
  updated: number;
  deleted: number;
  skipped: number;
  total_chunks: number;
  errors: string[];
  duration_seconds: number;
}

export interface StatsResult {
  total_files: number;
  total_chunks: number;
  collection_count: number;
  courses: Record<string, { files: number; chunks: number }>;
}

// ── Chat ─────────────────────────────────────────────────────

export async function* streamChat(
  question: string,
  course?: string,
  topK: number = 8
): AsyncGenerator<ChatEvent> {
  const response = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, course: course || null, top_k: topK }),
  });

  if (!response.ok) {
    throw new Error(`Chat request failed: ${response.statusText}`);
  }

  const reader = response.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (line.trim()) {
        try {
          const event: ChatEvent = JSON.parse(line);
          yield event;
        } catch {
          // Skip malformed lines
        }
      }
    }
  }

  // Process remaining buffer
  if (buffer.trim()) {
    try {
      yield JSON.parse(buffer);
    } catch {
      // Skip
    }
  }
}

// ── Documents ────────────────────────────────────────────────

export async function listDocuments(course?: string): Promise<{ documents: DocumentInfo[]; total: number }> {
  const params = course ? `?course=${encodeURIComponent(course)}` : "";
  const response = await fetch(`${API_BASE}/api/documents${params}`);
  return response.json();
}

export async function syncVault(): Promise<SyncResult> {
  const response = await fetch(`${API_BASE}/api/documents/sync`, { method: "POST" });
  return response.json();
}

export async function uploadFile(file: File, course: string, category: string = "Resources"): Promise<unknown> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("course", course);
  formData.append("category", category);

  const response = await fetch(`${API_BASE}/api/documents/upload`, {
    method: "POST",
    body: formData,
  });
  return response.json();
}

export async function getStats(): Promise<StatsResult> {
  const response = await fetch(`${API_BASE}/api/documents/stats`);
  return response.json();
}

// ── Courses ──────────────────────────────────────────────────

export async function listCourses(): Promise<{ courses: CourseInfo[] }> {
  const response = await fetch(`${API_BASE}/api/courses`);
  return response.json();
}

// ── Health ───────────────────────────────────────────────────

export async function checkHealth(): Promise<{ status: string; ollama: { status: string; models: string[] } }> {
  const response = await fetch(`${API_BASE}/api/health`);
  return response.json();
}
