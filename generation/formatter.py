def format_answer(answer, citations):
    formatted = []

    formatted.append(answer.strip())

    # sources
    if citations:
        for c in citations:
            line = f"- **{c['chunk_id']}**"
            if c.get("section"):
                line += f" (section: {c['section']})"
            if c.get("type"):
                line += f" — `{c['type']}`"
            formatted.append(line)

    return "\n".join(formatted)
