"""Retrieval over Apollo's own governance docs (Phase 4, Model C).

A deliberately small, dependency-light RAG corpus: the repo's markdown docs, split
into heading-anchored chunks and ranked by TF-IDF cosine similarity. No embeddings
service, no vector DB, no external calls — the corpus is small and curated, and
determinism matters more than recall here.

Only Apollo's authored docs are indexed. Raw GTD records and narratives are never
part of the corpus (licence + governance).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from training.config import REPO_ROOT

DOC_FILES = [
    "docs/intended_use.md",
    "docs/model_card.md",
    "docs/prediction_time_contract.md",
    "docs/feature_glossary.md",
]


@dataclass
class Chunk:
    source: str   # "model_card.md#evaluation"
    text: str


@dataclass
class Evidence:
    source: str
    snippet: str
    score: float


def _chunk_markdown(path: Path, max_chars: int = 700) -> list[Chunk]:
    """Split a markdown file into chunks anchored at ## / ### headings."""
    text = path.read_text(encoding="utf-8")
    chunks: list[Chunk] = []
    section = path.name
    buf: list[str] = []

    def flush():
        body = " ".join(" ".join(buf).split())
        if body:
            # Keep chunks bounded so one huge section can't dominate the index.
            for i in range(0, len(body), max_chars):
                chunks.append(Chunk(source=section, text=body[i:i + max_chars]))
        buf.clear()

    for line in text.splitlines():
        m = re.match(r"^#{2,3}\s+(.*)", line)
        if m:
            flush()
            slug = re.sub(r"[^a-z0-9]+", "-", m.group(1).lower()).strip("-")
            section = f"{path.name}#{slug}"
        else:
            buf.append(line)
    flush()
    return chunks


class Retriever:
    """TF-IDF retriever over the doc chunks. Built once at startup."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._matrix = self._vec.fit_transform([c.text for c in chunks])

    def retrieve(self, query: str, k: int = 4, *, min_score: float = 0.02) -> list[Evidence]:
        q = self._vec.transform([query])
        sims = cosine_similarity(q, self._matrix)[0]
        order = sims.argsort()[::-1][:k]
        out: list[Evidence] = []
        for i in order:
            score = float(sims[i])
            if score < min_score:
                continue
            out.append(Evidence(source=self.chunks[i].source,
                                snippet=self.chunks[i].text, score=round(score, 4)))
        return out


def build_retriever(root: Path | None = None) -> Retriever | None:
    """Index the doc corpus. Returns None if no docs are found."""
    root = root or REPO_ROOT
    chunks: list[Chunk] = []
    for rel in DOC_FILES:
        p = root / rel
        if p.exists():
            chunks.extend(_chunk_markdown(p))
    return Retriever(chunks) if chunks else None
