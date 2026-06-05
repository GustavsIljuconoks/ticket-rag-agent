from __future__ import annotations

from pathlib import Path

from .data_loader import create_task_document, task_metadata


class VectorStoreError(RuntimeError):
    """Raised when ChromaDB cannot be used."""


def require_chromadb():
    try:
        import chromadb
    except ImportError as exc:
        raise VectorStoreError(
            "chromadb is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return chromadb


class TicketVectorStore:
    def __init__(self, storage_dir: Path, collection_name: str):
        chromadb = require_chromadb()
        self.client = chromadb.PersistentClient(path=str(storage_dir))
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def index_tasks(self, tasks: list[dict], ollama_client) -> None:
        self.reset()

        ids = []
        documents = []
        metadatas = []
        embeddings = []

        for task in tasks:
            document = create_task_document(task)
            ids.append(task["task_id"])
            documents.append(document)
            metadatas.append(task_metadata(task))
            embeddings.append(ollama_client.embed(document))

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def count(self) -> int:
        return int(self.collection.count())

    def query(self, text: str, ollama_client, top_k: int) -> list[dict]:
        query_embedding = ollama_client.embed(text)
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        matches = []
        for index, task_id in enumerate(ids):
            distance = float(distances[index])
            similarity = max(0.0, min(1.0, 1.0 - distance))
            metadata = dict(metadatas[index])
            matches.append(
                {
                    "task_id": task_id,
                    "document": documents[index],
                    "metadata": metadata,
                    "distance": distance,
                    "similarity": similarity,
                }
            )

        return matches
