"""
Offline smoke test for the RAG engine. No API key or internet needed:
uses fake embeddings and a fake LLM, so it verifies chunking, caching,
hybrid search (BM25 + FAISS + RRF), query expansion and re-ranking logic.

Run:  python test_rag_offline.py
"""
import os
import tempfile
from pathlib import Path

from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag_core import KnowledgeBase, build_knowledge_tool, format_context

CASES = {
    "ecommerce": ("How long does express shipping take?", "Express shipping"),
    "education": ("What is the refund policy?", "Full refund"),
    "healthcare": ("Do I need to fast before a blood test?", "fasting"),
    "marketing": ("What is the minimum contract length?", "3-month minimum"),
    "realestate": ("How much is the down payment?", "down payment is 20 percent"),
    "restaurant": ("When are you open on Friday?", "midnight"),
}


class FakeMsg:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Pretends to be Gemini: returns rewrites for expansion, '0,1' for rerank."""
    def invoke(self, prompt):
        if "Rewrite the search query" in prompt:
            return FakeMsg("- cost and timing\n- how many days\n- delivery speed")
        return FakeMsg("0, 1")


def run():
    emb = DeterministicFakeEmbedding(size=64)
    for domain, (query, expected) in CASES.items():
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["RAG_MULTI_QUERY"] = "0"
            os.environ["RAG_RERANK"] = "0"
            kb = KnowledgeBase(domain, embeddings=emb, llm=FakeLLM(), index_dir=Path(tmp))
            assert kb.status == "built", kb.status
            docs = kb.retrieve(query)
            text = format_context(docs)
            assert expected.lower() in text.lower(), f"{domain}: '{expected}' not retrieved"

            # second start must use the cached index
            kb2 = KnowledgeBase(domain, embeddings=emb, llm=FakeLLM(), index_dir=Path(tmp))
            assert kb2.status == "loaded from cache", kb2.status

            # multi-query + rerank path must run and still return passages
            os.environ["RAG_MULTI_QUERY"] = "1"
            os.environ["RAG_RERANK"] = "1"
            kb3 = KnowledgeBase(domain, embeddings=emb, llm=FakeLLM(), index_dir=Path(tmp))
            assert kb3.retrieve(query), f"{domain}: empty result with multi-query/rerank"
        print(f"PASS  {domain:<11} {len(kb.chunks)} chunks")
    # retrieval chain + stuff documents chain (langchain_classic) through the agent tool
    os.environ["RAG_MULTI_QUERY"] = "0"
    os.environ["RAG_RERANK"] = "0"
    with tempfile.TemporaryDirectory() as tmp:
        llm = FakeListChatModel(responses=["Express shipping takes 1-2 business days. [store_policies.md]"])
        kb = KnowledgeBase("ecommerce", embeddings=emb, llm=llm, index_dir=Path(tmp))
        tool = build_knowledge_tool("ecommerce", kb=kb)
        out = tool.invoke({"query": "How long is express shipping?"})
        assert "1-2 business days" in out and "Sources: store_policies.md" in out, out
        os.environ["RAG_USE_CHAIN"] = "0"
        raw = build_knowledge_tool("ecommerce", kb=kb).invoke({"query": "express shipping"})
        assert "(source: store_policies.md)" in raw, raw
        os.environ["RAG_USE_CHAIN"] = "1"
    print("PASS  retrieval chain + stuff documents chain")
    print("\nAll RAG checks passed.")


if __name__ == "__main__":
    run()
