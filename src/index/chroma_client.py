from __future__ import annotations

import logging
from dataclasses import dataclass

import chromadb

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChromaTextResult:
    chunk_uid: str
    chunk_id: str
    score: float
    metadata: dict
    document: str


def get_client(persist_dir: str) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=persist_dir)


def get_text_collection(client: chromadb.PersistentClient, name: str = "text_chunks"):
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
