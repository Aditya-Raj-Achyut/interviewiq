"""Resume parsing, skill-gap analysis and a small RAG index (sentence-transformers + FAISS)."""
import re
from functools import lru_cache

SKILL_VOCAB = [
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "sql", "nosql",
    "mongodb", "postgresql", "mysql", "redis", "react", "node.js", "django", "flask", "fastapi",
    "streamlit", "html", "css", "git", "docker", "kubernetes", "aws", "azure", "gcp", "linux",
    "machine learning", "deep learning", "nlp", "computer vision", "llm", "rag", "langchain",
    "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy", "power bi", "tableau", "excel",
    "data analysis", "statistics", "etl", "spark", "hadoop", "airflow", "rest api", "microservices",
    "ci/cd", "agile", "scrum", "system design", "data structures", "algorithms", "oop",
    "generative ai", "prompt engineering", "mlops", "opencv", "transformers",
]


def extract_text(file) -> str:
    """Accepts a path or a file-like object (Streamlit upload) and returns the PDF text."""
    from pypdf import PdfReader

    reader = PdfReader(file)
    return "\n".join((page.extract_text() or "") for page in reader.pages).strip()


def find_skills(text: str) -> set:
    low = text.lower()
    found = set()
    for skill in SKILL_VOCAB:
        pattern = r"(?<![a-z0-9+#])" + re.escape(skill) + r"(?![a-z0-9+#])"
        if re.search(pattern, low):
            found.add(skill)
    return found


def skill_gap(resume_text: str, jd_text: str) -> dict:
    resume_skills = find_skills(resume_text)
    jd_skills = find_skills(jd_text)
    matched = sorted(resume_skills & jd_skills)
    missing = sorted(jd_skills - resume_skills)
    extra = sorted(resume_skills - jd_skills)
    match_pct = round(100 * len(matched) / len(jd_skills)) if jd_skills else 0
    return {"matched": matched, "missing": missing, "extra": extra, "match_pct": match_pct}


# ---------------- RAG ----------------
def chunk_text(text: str, size: int = 450, overlap: int = 80) -> list:
    words = text.split()
    chunks, step = [], max(size - overlap, 1)
    for i in range(0, len(words), step):
        piece = " ".join(words[i : i + size])
        if piece:
            chunks.append(piece)
    return chunks or [text]


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


class ResumeIndex:
    """Embeds resume chunks so we can fetch the parts most relevant to a query."""

    def __init__(self, text: str):
        self.chunks = chunk_text(text, size=120, overlap=30)
        self.index = None
        try:
            import faiss
            import numpy as np

            emb = _embedder().encode(self.chunks, normalize_embeddings=True)
            emb = np.asarray(emb, dtype="float32")
            self.index = faiss.IndexFlatIP(emb.shape[1])
            self.index.add(emb)
        except Exception:
            self.index = None  # fallback to keyword overlap

    def search(self, query: str, k: int = 3) -> list:
        k = min(k, len(self.chunks))
        if self.index is not None:
            import numpy as np

            q = _embedder().encode([query], normalize_embeddings=True)
            _, ids = self.index.search(np.asarray(q, dtype="float32"), k)
            return [self.chunks[i] for i in ids[0] if i >= 0]
        qw = set(query.lower().split())
        ranked = sorted(self.chunks, key=lambda c: -len(qw & set(c.lower().split())))
        return ranked[:k]
