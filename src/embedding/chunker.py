"""Text chunking for Vietnamese documents — recursive paragraph/sentence split."""

from typing import List, Dict


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks, respecting paragraph/sentence boundaries.

    Args:
        text: Input text (Vietnamese or mixed).
        chunk_size: Approximate word count per chunk.
        overlap: Words to carry over from previous chunk for context continuity.

    Returns:
        List of text chunk strings.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text] if text.strip() else []

    chunks: List[str] = []
    current_chunk = ""

    for paragraph in text.split("\n\n"):
        para_words = paragraph.split()
        current_words = current_chunk.split()

        if len(current_words) + len(para_words) <= chunk_size:
            # Fits in current chunk
            current_chunk = (current_chunk + "\n\n" + paragraph).lstrip()
        else:
            # Flush current chunk
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                # Carry overlap from tail of current chunk
                overlap_text = " ".join(current_chunk.split()[-overlap:])
                current_chunk = overlap_text + "\n\n" + paragraph
            else:
                # Single paragraph too long — split by sentences
                for sentence in paragraph.replace(". ", ".\n").split("\n"):
                    sent_words = sentence.split()
                    cur_words = current_chunk.split()
                    if len(cur_words) + len(sent_words) <= chunk_size:
                        current_chunk = (current_chunk + " " + sentence).lstrip()
                    else:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                            overlap_text = " ".join(current_chunk.split()[-overlap:])
                            current_chunk = overlap_text + " " + sentence
                        else:
                            current_chunk = sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def chunk_documents(documents: List[Dict]) -> List[Dict]:
    """Chunk all documents, preserving source metadata on each chunk.

    Args:
        documents: List of dicts with 'content', 'source', 'page', 'method' keys.

    Returns:
        List of chunk dicts with added 'metadata' field.
    """
    chunked: List[Dict] = []
    for doc in documents:
        text_chunks = chunk_text(doc["content"])
        for idx, chunk in enumerate(text_chunks):
            chunked.append({
                "content": chunk,
                "metadata": {
                    "source": doc.get("source", ""),
                    "page": doc.get("page", 0),
                    "chunk_idx": idx,
                    "total_chunks": len(text_chunks),
                    "method": doc.get("method", "unknown"),
                    "document_type": "educational",
                },
            })
    return chunked
