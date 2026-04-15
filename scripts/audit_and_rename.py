import os
import re
import base64
import shutil
from pathlib import Path
from openai import OpenAI
import fitz  # PyMuPDF

VAULT_DIR = "/Users/erjonmusa/Documents/StudyVault"
JUNK_DIR = "/Users/erjonmusa/Documents/StudyVault_JunkBin"
LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
MODEL_NAME = "google/gemma-4-26b-a4b"

PROMPT_TEXT = """What document is this? Give me the document type (exam/lecture/assignment), title, and date if visible. 
CRITICAL: ONLY return the final proposed file name without the .pdf extension. Do NOT include explanations, markdown formatting, or the .pdf extension.
Format it as: <Title if obvious, otherwise Type> - <Date or further info>. 
Examples: 
- Final Exam - April 2014
- Midterm - Winter 2022
- Week 01 - Lecture Notes"""

KNOWN_FILES = {
    "h15.pdf": "Final Exam - April 2014.pdf"
}

def needs_renaming(filename):
    if filename in KNOWN_FILES: 
        return True
    
    # Already well-formatted files containing a dash
    if " - " in filename: 
        return False
        
    name_no_ext = os.path.splitext(filename)[0].lower()
    
    # Leave explicit Labs and Assignments alone as they're generally okay
    if re.match(r"^(lab|assignment)s?\s*\d+.*$", name_no_ext):
        return False
        
    return True

def clean_filename(raw_resp):
    raw_resp = raw_resp.replace('.pdf', '')
    lines = raw_resp.strip().split('\n')
    for line in reversed(lines):
         line = line.strip().strip('*_"\'')
         if line and "based on" not in line.lower() and "i propose" not in line.lower() and "file name:" not in line.lower():
             # Basic cleanup for Windows/Mac bad path chars
             return re.sub(r'[/\\:*?"<>|]', '', line).strip()
    return re.sub(r'[/\\:*?"<>|]', '', raw_resp.strip()).strip()

def query_lmstudio(text=None, images_b64=None):
    client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key="lmstudio-link")
    
    messages = [
        {"role": "system", "content": PROMPT_TEXT}
    ]
    
    content_list = []
    
    if text:
        content_list.append({"type": "text", "text": f"Document text preview:\n{text[:1500]}"})
        
    if images_b64:
        content_list.append({"type": "text", "text": "Extract details from these scanned pages:"})
        for b64 in images_b64:
            content_list.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })
            
    messages.append({"role": "user", "content": content_list})

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=50
        )
        raw = response.choices[0].message.content.strip()
        return clean_filename(raw)
    except Exception as e:
        print(f"Error querying LM Studio: {e}")
        return None

def main():
    os.makedirs(JUNK_DIR, exist_ok=True)
    mappings = []
    
    for root, dirs, files in os.walk(VAULT_DIR):
        for file in files:
            filepath = os.path.join(root, file)
            
            # --- Check TXT / MD / Script Files (JUNK CATCHER) ---
            if file.lower().endswith(('.txt', '.md', '.sh', '.csv')):
                content = ""
                try:
                    with open(filepath, 'r') as f:
                        content = f.read(150)
                except Exception:
                    pass
                    
                code_keywords = ['CMake', '#include', 'def ', 'int main', 'Query:', 'Select ', 'script']
                junk_names = ["cmakelists", "fixdata", "query", "readme", "testing_doc", "design_doc", "marking", "test_", "assign", "marks", "toronto"]
                
                # If exact name match, substring name match, or contents contain code
                if any(bad in file.lower() for bad in junk_names) or any(k in content for k in code_keywords):
                    target_junk_path = os.path.join(JUNK_DIR, file)
                    
                    # Handle duplicate names in junk folder
                    counter = 2
                    while os.path.exists(target_junk_path):
                        name_no_ext, ext = os.path.splitext(file)
                        target_junk_path = os.path.join(JUNK_DIR, f"{name_no_ext} ({counter}){ext}")
                        counter += 1
                        
                    print(f"[\033[91mJUNK\033[0m] Moving project artifact out of vault: {file}")
                    shutil.move(filepath, target_junk_path)
                    continue
                
            # --- Check PDF Files ---
            if file.endswith(".pdf"):
                if not needs_renaming(file):
                    continue
                
                print(f"Processing: \033[96m{file}\033[0m")
                
                if file in KNOWN_FILES:
                    new_basename = KNOWN_FILES[file]
                    do_rename(filepath, root, file, new_basename, mappings)
                    continue
                
                # Use fitz to extract
                try:
                    doc = fitz.open(filepath)
                    text = ""
                    for i in range(min(2, len(doc))):
                        text += doc[i].get_text("text") + "\n"
                        
                    new_basename = None
                    
                    if len(text.strip()) > 50:
                        # Extractable Text
                        new_basename = query_lmstudio(text=text)
                    else:
                        # Scanned PDF - use Multimodal
                        print("  [INFO] Extracted text is empty. Falling back to Multimodal Gemma4...")
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
                        
                        if new_basename.lower() == file.lower():
                            print("  -> Original name is identical to proposed name. Skipping.")
                            continue
                            
                        do_rename(filepath, root, file, new_basename, mappings)
                    else:
                        print(f"  [ERROR] Could not generate a new name for {file}")
                
                except Exception as e:
                    print(f"  [ERROR] processing {file} - {e}")

    # Final summary
    print("\n" + "="*50)
    print("MAPPING SUMMARY (OLD -> NEW)")
    print("="*50)
    if not mappings:
        print("No files were renamed.")
    for old, new in mappings:
        print(f"{old} -> {new}")
        
def do_rename(filepath, root, old_file, new_basename, mappings):
    new_filepath = os.path.join(root, new_basename)
    
    counter = 2
    while os.path.exists(new_filepath) and new_filepath.lower() != filepath.lower():
        name_no_ext, ext = os.path.splitext(new_basename)
        new_filepath = os.path.join(root, f"{name_no_ext} ({counter}){ext}")
        counter += 1
        
    os.rename(filepath, new_filepath)
    mappings.append((old_file, os.path.basename(new_filepath)))
    print(f"  -> Renamed: \033[92m{os.path.basename(new_filepath)}\033[0m")

if __name__ == "__main__":
    main()
