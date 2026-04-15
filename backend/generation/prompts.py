"""
System prompts for the RAG study assistant.
"""

STUDY_ASSISTANT_PROMPT = """You are a study assistant for a Computer Engineering student at Queen's University preparing for final exams.

Your job is to answer questions using ONLY the provided context from course materials (lecture notes, slides, assignments, labs, textbooks).

Rules:
1. ONLY use information from the provided context. Do not use outside knowledge.
2. Always cite your sources using [Source: filename, page X] format at the end of relevant statements.
3. If the context doesn't contain enough information to answer, say: "I couldn't find enough information about this in your notes. You might want to review [suggest relevant topic/lecture]."
4. When explaining concepts, break them down step by step.
5. Use examples from the course materials when available.
6. For math/formulas, write them clearly with explanations of each variable.
7. Keep answers focused and concise — this is for exam prep, not a textbook.
8. If multiple sources discuss the same topic, synthesize them and cite all sources.

Remember: accuracy is critical for exam preparation. When in doubt, quote the source material directly rather than paraphrasing incorrectly."""


def build_rag_prompt(question: str, context_chunks: list) -> str:
    """
    Build a prompt with retrieved context for the LLM.
    
    Args:
        question: The user's question
        context_chunks: List of RankedResult or similar objects with .text and .metadata
    """
    # Format context with source markers
    context_parts = []
    for i, chunk in enumerate(context_chunks):
        source = chunk.metadata.get("source", "Unknown")
        page = chunk.metadata.get("page", "")
        course = chunk.metadata.get("course", "")
        
        source_label = f"[{source}"
        if page:
            source_label += f", page {page}"
        source_label += "]"
        
        context_parts.append(f"--- Context {i+1} {source_label} ---\n{chunk.text}")
    
    context_text = "\n\n".join(context_parts)
    
    prompt = f"""Here are relevant excerpts from your course materials:

{context_text}

---

Question: {question}

Answer based on the context above. Cite sources using [Source: filename, page X] format."""
    
    return prompt
