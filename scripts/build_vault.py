#!/usr/bin/env python3
import os
import shutil
import re
import fitz
import base64
from pathlib import Path
from audit_and_rename import query_lmstudio, needs_renaming, KNOWN_FILES

SOURCE_DIR = Path.home() / "Documents" / "Winter 2026"
VAULT_DIR = Path.home() / "Documents" / "StudyVault"

COURSE_NAMES = {
    "CMPE 223": "CMPE 223 - Software Specification",
    "ELEC 472": "ELEC 472 - Artificial Intelligence",
    "ELEC 477": "ELEC 477 - Distributed Systems",
}

SKIP_PATTERNS = {
    "node_modules", "build", "CMakeFiles", "cyclonedds",
    "cyclonedds-cxx", ".git", ".claude", "__MACOSX", ".next",
    ".DS_Store", "CAPSTONE", "venv", ".venv"
}

JUNK_FILE_KEYWORDS = ["cmakelists", "fixdata", "query", "readme", "testing_doc", "design_doc", "marking", "test_", "assign", "marks", "toronto"]

def detect_course(path: Path) -> str | None:
    path_str = str(path)
    for folder_name in COURSE_NAMES:
        if folder_name in path_str:
            return COURSE_NAMES[folder_name]
    return None

def detect_category(path: Path, filename: str) -> str:
    path_str = str(path).lower()
    filename_lower = filename.lower()

    if "exam" in path_str or "quiz" in path_str or "midterm" in path_str:
        return "Exams"
    if "lab" in path_str or "lab" in filename_lower:
        return "Labs"
    if "assignment" in path_str or path_str.endswith("/a1") or path_str.endswith("/a3"):
        return "Assignments"
    if "slide" in filename_lower or "lecture" in filename_lower or "notes" in filename_lower:
        return "Lectures"
    
    return "Resources"

def is_junk_file(file: str) -> bool:
    if file.lower().endswith(('.sh', '.csv')):
        return True
    
    if any(bad in file.lower() for bad in JUNK_FILE_KEYWORDS):
        return True
    
    return False

def get_beautiful_name(filepath: str, original_file: str) -> str:
    if original_file in KNOWN_FILES:
        return KNOWN_FILES[original_file]
        
    if not original_file.endswith(".pdf") or not needs_renaming(original_file):
        return original_file
        
    print(f"  [AI] Asking Gemma to rename: {original_file}")
    try:
        doc = fitz.open(filepath)
        text = ""
        for i in range(min(2, len(doc))):
            text += doc[i].get_text("text") + "\n"
            
        new_basename = None
        if len(text.strip()) > 50:
            new_basename = query_lmstudio(text=text)
        else:
            images_b64 = []
            for i in range(min(2, len(doc))):
                pix = doc[i].get_pixmap(dpi=150)
                b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
                images_b64.append(b64)
            if images_b64:
                new_basename = query_lmstudio(images_b64=images_b64)
        doc.close()
        
        if new_basename:
            if not new_basename.endswith('.pdf'):
                new_basename += '.pdf'
            return new_basename
            
    except Exception as e:
        print(f"  [Error] Failed to rename {original_file}: {e}")
        
    return original_file

def main():
    print(f"Wiping {VAULT_DIR} for a fresh build...")
    if VAULT_DIR.exists():
        shutil.rmtree(VAULT_DIR)
    
    VAULT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Pre-create course folders
    for course_full in COURSE_NAMES.values():
        for category in ["Lectures", "Assignments", "Labs", "Exams", "Resources"]:
            (VAULT_DIR / course_full / category).mkdir(parents=True, exist_ok=True)

    print("Scanning Original Directory...")
    
    copied = 0
    for root, dirs, files in os.walk(SOURCE_DIR):
        dirs[:] = [d for d in dirs if d not in SKIP_PATTERNS]
        
        root_path = Path(root)
        course = detect_course(root_path)
        
        if not course:
            continue
            
        for file in files:
            if file in SKIP_PATTERNS or file.startswith('.'):
                continue
                
            if is_junk_file(file):
                print(f"  [SKIPPED] Junk file filtered: {file}")
                continue
                
            source_filepath = root_path / file
            category = detect_category(root_path, file)
            
            clean_name = get_beautiful_name(str(source_filepath), file)
            
            dest_filepath = VAULT_DIR / course / category / clean_name
            
            # De-duplication fallback
            counter = 2
            while dest_filepath.exists():
                name_no_ext, ext = os.path.splitext(clean_name)
                dest_filepath = VAULT_DIR / course / category / f"{name_no_ext} ({counter}){ext}"
                counter += 1
                
            shutil.copy2(source_filepath, dest_filepath)
            print(f"  [COPIED] {course}/{category}/\033[92m{dest_filepath.name}\033[0m")
            copied += 1
            
    print(f"\nFresh Vault Rebuild Complete! Copied {copied} files.")
    print("Next step: Run `python scripts/force_ingest.py` to sync the database!")

if __name__ == "__main__":
    main()
