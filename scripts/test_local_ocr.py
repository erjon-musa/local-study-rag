"""
Smoke test: load LightOnOCR-2-1B on Apple Silicon (MPS) and OCR one image.

Run: PYTHONPATH=. python scripts/test_local_ocr.py [optional_image_or_pdf_path]

If no path is given, OCRs the first scanned PDF in StudyVault that has no
extractable text. Prints the model load time, OCR time, and extracted text.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def find_test_image() -> tuple[bytes, str]:
    """Find a scanned PDF page in the vault to test on, or fall back to a remote URL."""
    import fitz  # pymupdf

    vault = Path(os.getenv("VAULT_PATH", str(Path.home() / "Documents" / "StudyVault"))).expanduser()
    if not vault.exists():
        raise SystemExit(f"VAULT_PATH does not exist: {vault}")

    for pdf_path in vault.rglob("*.pdf"):
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(min(3, len(doc))):
                page = doc[page_num]
                if not page.get_text("text").strip() and page.get_images():
                    pixmap = page.get_pixmap(dpi=100)
                    jpeg_bytes = pixmap.tobytes("jpeg")
                    label = f"{pdf_path.name} (page {page_num + 1})"
                    doc.close()
                    return jpeg_bytes, label
            doc.close()
        except Exception:
            continue

    raise SystemExit("No scanned PDF pages found in vault to test on.")


def main():
    print("─" * 60)
    print("LightOnOCR-2-1B smoke test")
    print("─" * 60)

    # Step 1: find a test image
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if path.suffix.lower() == ".pdf":
            import fitz
            doc = fitz.open(path)
            page = doc[0]
            jpeg_bytes = page.get_pixmap(dpi=100).tobytes("jpeg")
            label = f"{path.name} (page 1)"
            doc.close()
        else:
            jpeg_bytes = path.read_bytes()
            label = path.name
    else:
        print("Searching vault for a scanned PDF page...")
        jpeg_bytes, label = find_test_image()

    print(f"Test image: {label} ({len(jpeg_bytes) / 1024:.1f} KB)")
    print()

    # Step 2: load model
    print("Loading LightOnOCR-2-1B (first run downloads ~2 GB)...")
    t0 = time.time()
    import torch
    from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float32 if device == "mps" else torch.bfloat16
    print(f"  device={device}  dtype={dtype}")

    model = LightOnOcrForConditionalGeneration.from_pretrained(
        "lightonai/LightOnOCR-2-1B", torch_dtype=dtype
    ).to(device)
    processor = LightOnOcrProcessor.from_pretrained("lightonai/LightOnOCR-2-1B")
    print(f"  loaded in {time.time() - t0:.1f}s")
    print()

    # Step 3: OCR
    print("Running OCR...")
    import io
    from PIL import Image

    pil_image = Image.open(io.BytesIO(jpeg_bytes))
    # Cap longest dimension to keep Pixtral's O(n²) attention from blowing up MPS
    MAX_DIM = 1024
    if max(pil_image.size) > MAX_DIM:
        scale = MAX_DIM / max(pil_image.size)
        new_size = (int(pil_image.size[0] * scale), int(pil_image.size[1] * scale))
        pil_image = pil_image.resize(new_size, Image.LANCZOS)
        print(f"  Resized image to {pil_image.size}")

    conversation = [{"role": "user", "content": [{"type": "image", "image": pil_image}]}]

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {
        k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
        for k, v in inputs.items()
    }

    if device == "mps":
        torch.mps.empty_cache()

    t0 = time.time()
    output_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    text = processor.decode(generated_ids, skip_special_tokens=True)
    elapsed = time.time() - t0

    print(f"  done in {elapsed:.1f}s ({len(text)} chars)")
    print()
    print("─" * 60)
    print("EXTRACTED TEXT")
    print("─" * 60)
    print(text[:2000])
    if len(text) > 2000:
        print(f"... [{len(text) - 2000} more chars]")


if __name__ == "__main__":
    main()
