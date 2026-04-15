"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { listCourses, type CourseInfo } from "@/lib/api";

interface SidebarProps {
  selectedCourse: string | null;
  onCourseSelect: (course: string | null) => void;
}

export default function Sidebar({
  selectedCourse,
  onCourseSelect,
}: SidebarProps) {
  const pathname = usePathname();
  const [courses, setCourses] = useState<CourseInfo[]>([]);

  useEffect(() => {
    listCourses()
      .then((data) => setCourses(data.courses))
      .catch(() => {});
  }, []);

  const courseColors: Record<string, string> = {
    "CMPE 223 - Software Specification": "var(--course-cmpe223)",
    "ELEC 472 - Artificial Intelligence": "var(--course-elec472)",
    "ELEC 477 - Distributed Systems": "var(--course-elec477)",
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="logo-icon">📚</div>
        Study RAG
      </div>

      <nav className="sidebar-nav">
        <Link
          href="/"
          className={`sidebar-link ${pathname === "/" ? "active" : ""}`}
        >
          <span className="link-icon">💬</span>
          Chat
        </Link>
        <Link
          href="/documents"
          className={`sidebar-link ${pathname === "/documents" ? "active" : ""}`}
        >
          <span className="link-icon">📄</span>
          Documents
        </Link>
      </nav>

      <div className="sidebar-section">
        <div className="sidebar-section-title">Filter by Course</div>

        <div
          className={`course-badge ${selectedCourse === null ? "selected" : ""}`}
          onClick={() => onCourseSelect(null)}
        >
          <span
            className="course-dot"
            style={{ background: "var(--text-muted)" }}
          />
          All Courses
        </div>

        {courses.map((course) => (
          <div
            key={course.name}
            className={`course-badge ${selectedCourse === course.name ? "selected" : ""}`}
            onClick={() => onCourseSelect(course.name)}
          >
            <span
              className="course-dot"
              style={{
                background:
                  courseColors[course.name] || "var(--accent-blue)",
              }}
            />
            <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {course.name.split(" - ")[0]}
            </span>
            <span className="course-count">{course.files}</span>
          </div>
        ))}
      </div>
    </aside>
  );
}
