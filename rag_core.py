"""
rag_core.py - Advanced RAG engine shared by all Zuvayra Systems agents.

Pipeline
--------
1. Ingest      : knowledge/<domain>/*.md|*.txt  (UTF-8)
2. Chunking    : Markdown-header-aware split + recursive split, every chunk
                 is prefixed with its section path ("Returns > Refunds")
3. Index       : FAISS (semantic) + BM25 (keyword). FAISS index is cached on
                 disk and rebuilt automatically when the knowledge files change
4. Retrieval   : multi-query expansion (LLM) -> semantic + keyword search for
                 every query -> Reciprocal Rank Fusion (hybrid)
5. Re-ranking  : LLM picks the most relevant passages from the fused pool
6. Chain       : create_retrieval_chain + create_stuff_documents_chain (langchain_classic)
                 stuff the passages into a strict grounded prompt -> cited answer
                 (falls back to raw passages if RAG_USE_CHAIN=0 or the chain fails)

Every LLM step falls back gracefully, so retrieval never crashes the agent.

Environment variables (all optional)
------------------------------------
GOOGLE_API_KEY      required
GEMINI_MODEL        default: gemini-3.5-flash
EMBEDDING_MODEL     default: models/gemini-embedding-001
RAG_CHUNK_SIZE      default: 600
RAG_CHUNK_OVERLAP   default: 80
RAG_TOP_K           default: 4    passages returned to the agent
RAG_FETCH_K         default: 8    candidates per search before fusion/rerank
RAG_MULTI_QUERY     default: 0    set 1 for query expansion (+1 LLM call, slower)
RAG_RERANK          default: 0    set 1 for LLM re-ranking (+1 LLM call, slower)
RAG_USE_CHAIN       default: 1    retrieval chain + stuff-documents chain (grounded answer,
                                  +1 LLM call). Set 0 to return raw passages (faster)
RAG_DEBUG           default: 0    set 1 to print retrieval timings
"""

import hashlib
import os
import re
import time
from pathlib import Path
from typing import Any, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from rank_bm25 import BM25Okapi

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
INDEX_DIR = BASE_DIR / ".rag_index"

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


# ---------------------------------------------------------------- helpers

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _env_flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9$%]+", text.lower())


def _msg_text(msg) -> str:
    """Get plain text from a LangChain message across versions."""
    text = getattr(msg, "text", None)
    if callable(text):
        text = text()
    if isinstance(text, str) and text:
        return text
    content = getattr(msg, "content", "")
    if isinstance(content, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return str(content)


def _rrf(rank_lists: List[List[Document]], k: int = 60) -> List[Document]:
    """Reciprocal Rank Fusion: merge several ranked lists into one."""
    scores, docs = {}, {}
    for ranked in rank_lists:
        for rank, doc in enumerate(ranked):
            cid = doc.metadata["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            docs[cid] = doc
    return [docs[c] for c in sorted(scores, key=scores.get, reverse=True)]


def _default_embeddings():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found. Add it to your .env file.")
    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001"),
        google_api_key=api_key,
    )


def _default_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


# ---------------------------------------------------------- knowledge base

class KnowledgeBase:
    def __init__(self, domain: str, embeddings=None, llm=None,
                 knowledge_dir: Optional[Path] = None,
                 index_dir: Optional[Path] = None):
        self.domain = domain
        self.knowledge_dir = Path(knowledge_dir or KNOWLEDGE_DIR / domain)
        self.index_dir = Path(index_dir or INDEX_DIR / domain)
        self.chunk_size = _env_int("RAG_CHUNK_SIZE", 600)
        self.chunk_overlap = _env_int("RAG_CHUNK_OVERLAP", 80)
        self.top_k = _env_int("RAG_TOP_K", 4)
        self.fetch_k = _env_int("RAG_FETCH_K", 8)
        self.multi_query = _env_flag("RAG_MULTI_QUERY", "0")
        self.rerank = _env_flag("RAG_RERANK", "0")
        self.debug = _env_flag("RAG_DEBUG", "0")
        self._cache = {}

        self._embeddings = embeddings
        self._llm = llm

        self.sources = self._read_sources()
        self.chunks = self._build_chunks()
        self.status = "empty"
        self.vector_store = self._load_or_build_index()
        self.bm25 = BM25Okapi([_tokenize(c.page_content) for c in self.chunks])

    # -- lazy model getters
    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = _default_embeddings()
        return self._embeddings

    @property
    def llm(self):
        if self._llm is None:
            self._llm = _default_llm()
        return self._llm

    # -- ingestion
    def _read_sources(self):
        if not self.knowledge_dir.is_dir():
            raise FileNotFoundError(
                f"Knowledge folder not found: {self.knowledge_dir}\n"
                f"Create it and add .md or .txt files."
            )
        files = sorted(
            p for p in self.knowledge_dir.iterdir()
            if p.suffix.lower() in (".md", ".txt")
        )
        if not files:
            raise FileNotFoundError(
                f"No .md/.txt files in {self.knowledge_dir}"
            )
        return [(p.name, p.read_text(encoding="utf-8")) for p in files]

    def _build_chunks(self) -> List[Document]:
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=HEADERS, strip_headers=True
        )
        rec_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        chunks: List[Document] = []
        for name, text in self.sources:
            idx = 0
            for section in md_splitter.split_text(text):
                path = " > ".join(
                    section.metadata[h] for _, h in HEADERS if h in section.metadata
                )
                for piece in rec_splitter.split_text(section.page_content):
                    content = f"[{path}]\n{piece}" if path else piece
                    chunks.append(Document(
                        page_content=content,
                        metadata={"source": name, "section": path,
                                  "chunk_id": f"{name}#{idx}"},
                    ))
                    idx += 1
        return chunks

    # -- index (cached on disk, auto-rebuild on change)
    def _fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(f"{self.chunk_size}|{self.chunk_overlap}|"
                 f"{os.getenv('EMBEDDING_MODEL', '')}".encode())
        for name, text in self.sources:
            h.update(name.encode())
            h.update(text.encode("utf-8"))
        return h.hexdigest()

    def _load_or_build_index(self):
        fp_file = self.index_dir / "fingerprint.txt"
        fingerprint = self._fingerprint()
        if fp_file.exists() and fp_file.read_text().strip() == fingerprint:
            try:
                store = FAISS.load_local(
                    str(self.index_dir), self.embeddings,
                    allow_dangerous_deserialization=True,  # our own local index
                )
                self.status = "loaded from cache"
                return store
            except Exception:
                pass  # corrupt cache -> rebuild
        store = FAISS.from_documents(self.chunks, self.embeddings)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        store.save_local(str(self.index_dir))
        fp_file.write_text(fingerprint)
        self.status = "built"
        return store

    # -- retrieval steps
    def _bm25_search(self, query: str, k: int) -> List[Document]:
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i] for i in order[:k] if scores[i] > 0]

    def _expand_query(self, query: str) -> List[str]:
        prompt = (
            "Rewrite the search query below in 3 different ways (synonyms, "
            "different wording, more specific). Output ONLY the 3 rewrites, "
            "one per line, no numbering.\n\nQuery: " + query
        )
        try:
            lines = _msg_text(self.llm.invoke(prompt)).splitlines()
            out = [re.sub(r"^[\s\-*\d.)]+", "", ln).strip() for ln in lines]
            return [q for q in out if q and q.lower() != query.lower()][:3]
        except Exception:
            return []

    def _rerank(self, query: str, docs: List[Document], k: int) -> List[Document]:
        listing = "\n\n".join(
            f"[{i}] {d.page_content[:400]}" for i, d in enumerate(docs)
        )
        prompt = (
            f"Question: {query}\n\nPassages:\n{listing}\n\n"
            f"Pick up to {k} passages that best help answer the question. "
            "Reply ONLY with their numbers, best first, comma-separated."
        )
        try:
            nums = [int(n) for n in re.findall(r"\d+", _msg_text(self.llm.invoke(prompt)))]
        except Exception:
            return docs
        chosen, seen = [], set()
        for n in nums:
            if 0 <= n < len(docs) and n not in seen:
                seen.add(n)
                chosen.append(docs[n])
        if not chosen:
            return docs
        chosen += [d for i, d in enumerate(docs) if i not in seen]
        return chosen

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        k = k or self.top_k
        cache_key = (query.strip().lower(), k)
        if cache_key in self._cache:
            return self._cache[cache_key]

        t0 = time.time()
        queries = [query]
        if self.multi_query:
            queries += self._expand_query(query)
        t1 = time.time()

        rank_lists = []
        for q in queries:
            rank_lists.append(self.vector_store.similarity_search(q, k=self.fetch_k))
            rank_lists.append(self._bm25_search(q, self.fetch_k))
        pool = _rrf(rank_lists)[: self.fetch_k]
        t2 = time.time()

        if self.rerank and len(pool) > k:
            pool = self._rerank(query, pool, k)
        t3 = time.time()

        if self.debug:
            print(f"[RAG] expand {t1 - t0:.2f}s | search {t2 - t1:.2f}s | "
                  f"rerank {t3 - t2:.2f}s | total {t3 - t0:.2f}s")
        result = pool[:k]
        if len(self._cache) > 200:
            self._cache.clear()
        self._cache[cache_key] = result
        return result


def format_context(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {d.metadata['source']})\n{d.page_content}"
        for i, d in enumerate(docs, 1)
    )


# ------------------------------------------------------------ chains

class HybridRetriever(BaseRetriever):
    """LangChain retriever wrapper around KnowledgeBase.retrieve (hybrid search)."""
    kb: Any
    k: int = 4

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        return self.kb.retrieve(query, self.k)


GROUNDED_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer customer questions for a business using ONLY the context below.\n"
     "- If the context does not contain the answer, say you do not have that information.\n"
     "- Never invent prices, times, policies or numbers.\n"
     "- Be concise (2-4 sentences) and mention the source file name in brackets.\n\n"
     "Context:\n{context}"),
    ("human", "{input}"),
])

DOC_PROMPT = PromptTemplate.from_template("[source: {source}]\n{page_content}")


def build_rag_chain(kb: KnowledgeBase):
    """retriever -> stuff documents chain -> {'input', 'context', 'answer'}"""
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain

    combine_docs_chain = create_stuff_documents_chain(
        kb.llm, GROUNDED_PROMPT, document_prompt=DOC_PROMPT
    )
    return create_retrieval_chain(HybridRetriever(kb=kb, k=kb.top_k), combine_docs_chain)


# ------------------------------------------------------------ agent tool

def build_knowledge_tool(domain: str, kb: Optional[KnowledgeBase] = None):
    """Build the `search_knowledge_base` tool for one agent domain."""
    kb = kb or KnowledgeBase(domain)
    print(f"[RAG] {domain}: {len(kb.chunks)} chunks from "
          f"{len(kb.sources)} file(s), index {kb.status}")

    chain = None
    if _env_flag("RAG_USE_CHAIN", "1"):
        try:
            chain = build_rag_chain(kb)
            print(f"[RAG] {domain}: retrieval chain enabled")
        except Exception as e:
            print(f"[RAG] {domain}: chain unavailable ({e}); using raw passages")

    @tool
    def search_knowledge_base(query: str) -> str:
        """
        Search the business knowledge base for policies, FAQs, procedures,
        timings, terms and conditions. Returns a grounded answer with sources.
        Use this for any question about how the business works.
        Input: a clear, self-contained search query.
        """
        if chain is not None:
            try:
                out = chain.invoke({"input": query})
                answer = (out.get("answer") or "").strip()
                sources = sorted({d.metadata["source"] for d in out.get("context", [])})
                if answer:
                    return answer + (f"\n\n(Sources: {', '.join(sources)})" if sources else "")
            except Exception as e:
                print(f"[RAG] chain failed, falling back to passages: {e}")
        try:
            docs = kb.retrieve(query)
        except Exception as e:
            return f"Knowledge base search failed: {e}"
        if not docs:
            return "No relevant information found in the knowledge base."
        return format_context(docs)

    return search_knowledge_base
