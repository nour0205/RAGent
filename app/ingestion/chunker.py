import re


def base_chunks(text: str, max_chars: int = 500) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks


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