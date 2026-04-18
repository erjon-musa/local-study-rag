"use client";

import { useEffect, useRef, useState } from "react";
import { listCourses, type CourseInfo } from "@/lib/api";

interface CourseSelectorPillProps {
  /** null = All Courses */
  selectedCourse: string | null;
  /**
   * Called with the newly-chosen course (or null for All Courses).
   * Parent is responsible for: aborting in-flight requests, bumping the
   * generation counter, clearing messages, and showing the toast.
   */
  onChange: (course: string | null) => void;
}

export default function CourseSelectorPill({
  selectedCourse,
  onChange,
}: CourseSelectorPillProps) {
  const [courses, setCourses] = useState<CourseInfo[]>([]);
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    listCourses()
      .then((data) => setCourses(data.courses))
      .catch(() => {});
  }, []);

  // Close dropdown on outside click.
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  const label = selectedCourse
    ? selectedCourse.split(" - ").slice(0, 2).join(" — ")
    : "All Courses";

  const handlePick = (course: string | null) => {
    setOpen(false);
    if (course === selectedCourse) return; // no-op: don't nuke messages
    onChange(course);
  };

  return (
    <div className="course-pill-root" ref={rootRef}>
      <button
        className="course-pill"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <span className="course-pill-prefix">Chatting about:</span>
        <span className="course-pill-label">{label}</span>
        <span className={`course-pill-caret${open ? " open" : ""}`} aria-hidden="true">
          ▾
        </span>
      </button>

      {open && (
        <div className="course-pill-menu" role="listbox">
          <button
            className={`course-pill-option${selectedCourse === null ? " selected" : ""}`}
            role="option"
            aria-selected={selectedCourse === null}
            onClick={() => handlePick(null)}
          >
            All Courses
          </button>
          {courses.map((c) => (
            <button
              key={c.name}
              className={`course-pill-option${selectedCourse === c.name ? " selected" : ""}`}
              role="option"
              aria-selected={selectedCourse === c.name}
              onClick={() => handlePick(c.name)}
            >
              {c.name}
              <span className="course-pill-option-count">{c.files}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
