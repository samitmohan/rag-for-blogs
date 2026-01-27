SYSTEM_PROMPT = (
    "You are a retrieval-augmented assistant.\n"
    "Answer the user's question using ONLY the information provided in the context.\n"
    "Do NOT use prior knowledge or make assumptions.\n"
    "If the context does not contain enough information to answer the question, "
    "say exactly: \"I don’t know based on the provided information.\" \n\n"
    "Guidelines:\n"
    "- Be concise and precise.\n"
    "- Do NOT include section headers like 'Answer' or 'Sources'.\n"
    "- Do NOT repeat the question.\n"
    "- Do NOT fabricate facts, code, or citations.\n"
    "- If code is provided in the context and the question asks for code, return it verbatim.\n"
    "- If the question asks for explanation, prefer text context over code.\n"
)

def build_prompt(query: str, retrieved_chunks):
    context_blocks = []

    for idx, chunk in enumerate(retrieved_chunks, start=1):
        chunk_type = chunk["metadata"].get("type", "text")
        text = chunk["text"].strip()
        chunk_id = chunk["id"]

        if not text:
            continue

        header = f"[{chunk_type.upper()} | source={chunk_id}]"

        context_blocks.append(
            f"{header}\n{text}"
        )

    context = "\n\n".join(context_blocks)

    user_prompt = (
        f"Context:\n"
        f"{context}\n\n"
        f"Question:\n"
        f"{query}"
    )

    return {
        "system": SYSTEM_PROMPT,
        "user": user_prompt
    }
