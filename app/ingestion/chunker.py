def base_chunks(text: str) -> list[str]:
    return [line.strip() for line in text.split("\n") if line.strip()]
def apply_overlap(chunks: list[str], overlap: int = 1) -> list[str]:
    if overlap <= 0:
        return chunks

    overlapped = []

    for i, chunk in enumerate(chunks):
        start = max(0, i - overlap)
        context_chunks = chunks[start:i]

        if context_chunks:
            overlapped.append(" ".join(context_chunks + [chunk]))
        else:
            overlapped.append(chunk)

    return overlapped

