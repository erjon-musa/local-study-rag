/**
 * Streaming citation parser.
 *
 * While the LLM streams tokens, inline citation markers of the form `[N]` (1–2
 * digits) may arrive split across token boundaries — e.g. one chunk ends with
 * `"...heuristic "` and the next begins with `"[1"`, with `"]"` coming in a
 * third chunk. We buffer up to 8 characters of pending text so markers can
 * straddle boundaries without getting flushed as literal text.
 *
 * Rules:
 *  - If the buffer contains no '[' → flush everything to display.
 *  - If it contains '[' but no closing ']' yet AND buffer length < 8 → hold.
 *  - If a complete `[N]` or `[NN]` appears at the position of the earliest
 *    '[' → emit any text before it as a "text" segment, then emit the citation
 *    as a "citation" segment.
 *  - If buffer hits 8 chars with a '[' but no valid 1–2 digit match (e.g. a
 *    stray bracket, or a 3+ digit number like `[2024]`) → flush the FIRST char
 *    as literal and keep trying from position 1. This bounds the hold and
 *    never loses text.
 *
 * Usage:
 *   const parser = createCitationParser();
 *   parser.push(token) → Segment[]
 *   parser.flush()     → Segment[]   (call once at end of stream)
 */

export type Segment =
  | { type: "text"; value: string }
  | { type: "citation"; value: number };

/** Matches exactly `[N]` or `[NN]` — not `[2024]`, not `[]`, not `[abc]`. */
const CITATION_RE = /^\[(\d{1,2})\]/;

const MAX_BUFFER = 8;

export interface CitationParser {
  push(token: string): Segment[];
  flush(): Segment[];
}

export function createCitationParser(): CitationParser {
  let buffer = "";

  /**
   * Consume as much of `buffer` as possible, returning the segments produced.
   * Stops when the remaining buffer might still be part of an in-flight marker.
   */
  const drain = (allowHold: boolean): Segment[] => {
    const out: Segment[] = [];

    while (buffer.length > 0) {
      const openIdx = buffer.indexOf("[");

      // No bracket at all → safe to flush everything.
      if (openIdx === -1) {
        out.push({ type: "text", value: buffer });
        buffer = "";
        break;
      }

      // Flush any prefix before the bracket as literal text.
      if (openIdx > 0) {
        out.push({ type: "text", value: buffer.slice(0, openIdx) });
        buffer = buffer.slice(openIdx);
        continue;
      }

      // buffer[0] === '[' — try to match a complete citation here.
      const m = CITATION_RE.exec(buffer);
      if (m) {
        out.push({ type: "citation", value: parseInt(m[1], 10) });
        buffer = buffer.slice(m[0].length);
        continue;
      }

      // No complete match yet. Could be still arriving (`"["`, `"[1"`, `"[12"`)
      // OR invalid (`"[2"` followed by more digits like `"[2024]"`, or `"[a"`).
      // Hold only if the buffer is short AND could still become valid.
      if (allowHold && buffer.length < MAX_BUFFER) {
        // Heuristic: hold if what's after '[' so far is digits-then-maybe-']'.
        // This lets `[`, `[1`, `[12`, `[1]` (already handled above) all wait.
        // But `[a` or `[2024` (3+ digits with no ]) → flush '[' as literal and
        // continue — they can never become valid 1-2 digit citations.
        const rest = buffer.slice(1);
        const looksLikeCitation = /^\d{0,2}\]?$/.test(rest);
        if (looksLikeCitation) {
          break; // wait for more tokens
        }
        // Can't become a valid [N]/[NN] — flush the '[' as literal text.
        out.push({ type: "text", value: "[" });
        buffer = buffer.slice(1);
        continue;
      }

      // Buffer full (>= 8 chars) or flush mode — give up on this bracket.
      // Flush the '[' literally and keep scanning from position 1.
      out.push({ type: "text", value: "[" });
      buffer = buffer.slice(1);
    }

    return mergeAdjacentText(out);
  };

  return {
    push(token: string): Segment[] {
      buffer += token;
      return drain(true);
    },
    flush(): Segment[] {
      return drain(false);
    },
  };
}

/**
 * Coalesce runs of "text" segments so rendering doesn't emit 1-char nodes.
 * Citations are kept separate so React can render each as its own <sup>.
 */
function mergeAdjacentText(segs: Segment[]): Segment[] {
  if (segs.length < 2) return segs;
  const out: Segment[] = [];
  for (const s of segs) {
    const last = out[out.length - 1];
    if (s.type === "text" && last && last.type === "text") {
      last.value += s.value;
    } else {
      out.push({ ...s });
    }
  }
  return out;
}

/**
 * Convenience: parse an already-complete string into segments in one shot.
 * Used when rendering a finalized assistant turn from history (no streaming).
 */
export function parseCitations(text: string): Segment[] {
  const parser = createCitationParser();
  const a = parser.push(text);
  const b = parser.flush();
  return mergeAdjacentText([...a, ...b]);
}
