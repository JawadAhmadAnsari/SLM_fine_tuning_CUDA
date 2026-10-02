# src/kb.py
"""
Markdown knowledge base helpers, with no heavy dependencies.

The knowledge base is split into one chunk per "## " section. The RAG index
(src/rag.py) and the dataset builder (scripts/build_pakwheels_dataset.py) both
use this function, so the context the model sees in training is exactly the
text it retrieves at inference time.
"""

import re

SOURCE_RE = re.compile(r"^Source:\s*(\S+)\s*$", re.M)


def split_markdown_sections(text: str) -> list:
    """
    Returns [{"title", "content", "source"}] for every "## " section.
    Text before the first "## " (the file header) is ignored.
    `content` includes the "## Title" line so the chunk is self-describing.
    """
    sections = []
    for block in re.split(r"(?m)^## ", text)[1:]:
        title, _, body = block.partition("\n")
        content = f"## {title.strip()}\n{body.strip()}"
        source = SOURCE_RE.search(body)
        sections.append({
            "title": title.strip(),
            "content": content,
            "source": source.group(1) if source else None,
        })
    return sections


def load_sections(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return split_markdown_sections(f.read())


def sections_by_title(path: str) -> dict:
    return {s["title"]: s["content"] for s in load_sections(path)}
