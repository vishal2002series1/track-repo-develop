# src/memory/episodic.py
"""
Episodic memory for the Aeon router.

Stores one record per /api/chat turn so the planner can:
  - reuse recent context within the same session ("sticky routing")
  - learn from semantically similar past turns ("personalization")

Phase-1 backend: persistent ChromaDB at data_local/episodic_chroma/
Embedder:        local HuggingFace MiniLM (same model used elsewhere in the repo)

The class is intentionally backend-agnostic so Phase-N can swap in a
pgvector-backed implementation without touching any caller.
"""

from __future__ import annotations

import os
import json
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings

# --- Paths -------------------------------------------------------------------
# Keep episodic storage SEPARATE from the existing RAG chroma_db so it can be
# wiped / migrated independently.
_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
DEFAULT_EPISODIC_DIR = os.path.join(_BASE_DIR, 'data_local', 'episodic_chroma')
DEFAULT_MODEL_PATH = os.path.join(_BASE_DIR, 'local_embedding_model')

# Single global embedder instance — loading MiniLM is slow (~1-2s), so we want
# exactly one per process. Lazily initialized on first use.
_embedder: Optional[HuggingFaceEmbeddings] = None


def _get_embedder(model_path: str = DEFAULT_MODEL_PATH) -> HuggingFaceEmbeddings:
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(model_name=model_path)
    return _embedder


# --- Helpers -----------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _summarize(text: str, limit: int = 500) -> str:
    """Trim a long answer down to something safe to store + embed."""
    if not text:
        return ""
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _episode_id(session_id: str, turn_idx: int) -> str:
    return f"{session_id}:{turn_idx}:{uuid.uuid4().hex[:8]}"


# --- Episodic memory class ---------------------------------------------------
class EpisodicMemory:
    """
    Chroma-backed episodic memory store.

    Usage:
        mem = EpisodicMemory()
        mem.record(session_id="s1", turn_idx=0, user_prompt="...", final_answer="...",
                   plan={"mode": "single", "steps": [{"workflow_id": "WF_002"}]},
                   confidence="high", tenant_id="default", user_id=None)

        mem.recent(session_id="s1", k=3)
        mem.similar(prompt="...", k=3, tenant_id="default", user_id=None)
    """

    def __init__(
        self,
        persist_dir: str = DEFAULT_EPISODIC_DIR,
        collection_name: str = "aeon_episodes",
        model_path: str = DEFAULT_MODEL_PATH,
    ):
        os.makedirs(persist_dir, exist_ok=True)

        # Disable Chroma's anonymous telemetry — matches the rest of your repo.
        os.environ.setdefault("CHROMA_TELEMETRY_IMPL", "None")

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        # get_or_create_collection is idempotent — safe across restarts and replicas.
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # match MiniLM's similarity_fn_name
        )
        self._embedder = _get_embedder(model_path)

    # -- Write ---------------------------------------------------------------
    def record(
        self,
        *,
        session_id: str,
        turn_idx: int,
        user_prompt: str,
        final_answer: str,
        plan: Dict[str, Any],
        confidence: str = "medium",
        tenant_id: str = "default",
        user_id: Optional[str] = None,
    ) -> str:
        """Persist one episode. Returns the new episode_id."""
        episode_id = _episode_id(session_id, turn_idx)
        answer_summary = _summarize(final_answer)
        document = f"{user_prompt}\n---\n{answer_summary}"

        # One embed call per turn; cheap with MiniLM.
        embedding = self._embedder.embed_query(document)

        # Chroma metadata values must be primitives — JSON-encode the plan.
        workflow_ids = ",".join(
            (s.get("workflow_id") or "") for s in plan.get("steps", [])
        )

        metadata = {
            "tenant_id": tenant_id,
            "user_id": user_id if user_id is not None else "",
            "session_id": session_id,
            "turn_idx": int(turn_idx),
            "ts_iso": _now_iso(),
            "plan_json": json.dumps(plan, ensure_ascii=False),
            "workflow_ids": workflow_ids,
            "confidence": confidence,
        }

        self._collection.add(
            ids=[episode_id],
            documents=[document],
            embeddings=[embedding],
            metadatas=[metadata],
        )
        return episode_id

    # -- Read: recent session context ----------------------------------------
    def recent(self, session_id: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Return the most recent episodes for a session, newest first.

        Used by the planner to resolve follow-ups ("...and what about Chen?")
        without making the user repeat themselves.
        """
        # Chroma's .get() supports metadata filters but not ordering, so we
        # over-fetch a small page and sort in Python. Sessions are short
        # (<<100 turns); this is fine.
        res = self._collection.get(
            where={"session_id": session_id},
            include=["documents", "metadatas"],
            limit=200,
        )
        rows = self._zip_rows(res)
        rows.sort(key=lambda r: r["metadata"].get("turn_idx", 0), reverse=True)
        return rows[:k]

    # -- Read: semantically similar episodes ---------------------------------
    def similar(
        self,
        prompt: str,
        k: int = 3,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        exclude_session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return top-K semantically similar past episodes.

        Scoping rule (tightest available wins):
          - if user_id given   → filter by user_id  (most personal)
          - elif tenant_id     → filter by tenant_id (org-wide learning)
          - else               → no scope filter (dev/demo only)

        exclude_session_id lets the planner avoid re-surfacing the current
        session's own episodes via the similarity path (we already have those
        from .recent()).
        """
        where_clauses: List[Dict[str, Any]] = []
        if user_id:
            where_clauses.append({"user_id": user_id})
        elif tenant_id:
            where_clauses.append({"tenant_id": tenant_id})
        if exclude_session_id:
            where_clauses.append({"session_id": {"$ne": exclude_session_id}})

        where: Optional[Dict[str, Any]] = None
        if len(where_clauses) == 1:
            where = where_clauses[0]
        elif len(where_clauses) > 1:
            where = {"$and": where_clauses}

        embedding = self._embedder.embed_query(prompt)
        res = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return self._zip_query_rows(res)

    # -- Maintenance ---------------------------------------------------------
    def count(self) -> int:
        return self._collection.count()

    def clear(self) -> None:
        """Test helper: wipe the collection (does not delete the directory)."""
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name,
            metadata={"hnsw:space": "cosine"},
        )

    # -- Internal: normalize Chroma return shapes -----------------------------
    @staticmethod
    def _zip_rows(res: Dict[str, Any]) -> List[Dict[str, Any]]:
        ids = res.get("ids", []) or []
        docs = res.get("documents", []) or []
        metas = res.get("metadatas", []) or []
        return [
            {"id": i, "document": d, "metadata": m}
            for i, d, m in zip(ids, docs, metas)
        ]

    @staticmethod
    def _zip_query_rows(res: Dict[str, Any]) -> List[Dict[str, Any]]:
        # .query() returns nested lists keyed by query index (we only send 1)
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        return [
            {"id": i, "document": d, "metadata": m, "distance": float(dist)}
            for i, d, m, dist in zip(ids, docs, metas, dists)
        ]