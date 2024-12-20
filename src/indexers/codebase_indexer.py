import logging
from pathlib import Path
from typing import Iterator, Optional

from pathspec import PathSpec
from qdrant_client import QdrantClient, models

from src.processors.document_processor import DocumentProcessor

logger = logging.getLogger("main_logger")


class CodebaseIndexer:
    """indexer using Qdrant and dual embeddings"""

    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.processor = DocumentProcessor()
        self.client = QdrantClient(":memory:")  # Use in-memory for testing
        self.collection_name = "codebase"
        self.gitignore_spec = self._load_gitignore()

        # Initialize Qdrant collection
        self.client.create_collection(
            self.collection_name,
            vectors_config={
                "text": models.VectorParams(
                    size=384,  # MiniLM dimension
                    distance=models.Distance.COSINE,
                ),
                "code": models.VectorParams(
                    size=768,  # Jina dimension
                    distance=models.Distance.COSINE,
                ),
            },
        )

    def _load_gitignore(self) -> Optional[PathSpec]:
        """Loads and parses the .gitignore file if it exists"""
        gitignore_path = self.project_path / ".gitignore"
        if gitignore_path.is_file():
            with gitignore_path.open("r") as f:
                patterns = f.readlines()
            return PathSpec.from_lines("gitwildmatch", patterns)
        return None

    def index_project(self):
        """Indexes the project using dual embeddings"""
        points = []
        for file_path in self._get_project_files():
            try:
                with open(file_path, "r") as f:
                    content = f.read()

                metadata = {
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "module": file_path.parent.name,
                }

                nlp_embedding, code_embedding = self.processor.process_document(
                    content, metadata
                )

                points.append(
                    models.PointStruct(
                        id=len(points),
                        vector={
                            "text": nlp_embedding.tolist(),
                            "code": code_embedding.tolist(),
                        },
                        payload={"content": content, "metadata": metadata},
                    )
                )

                if len(points) >= 100:  # Batch upload
                    self.client.upload_points(self.collection_name, points)
                    points = []

            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {str(e)}")

        if points:  # Upload remaining points
            self.client.upload_points(self.collection_name, points)

    def _get_project_files(self) -> Iterator[Path]:
        """Yields all supported files in the project"""
        for path in self.project_path.rglob("*"):
            if self.gitignore_spec and self.gitignore_spec.match_file(
                str(path.relative_to(self.project_path))
            ):
                continue
            if path.is_file() and self.processor.get_file_type(str(path)):
                yield path

    def search(self, query: str, limit: int = 5):
        """Search using both embeddings and combine results"""
        return self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=next(self.processor.nlp_model.query_embed(query)).tolist(),
                    using="text",
                    limit=limit,
                ),
                models.Prefetch(
                    query=next(self.processor.code_model.query_embed(query)).tolist(),
                    using="code",
                    limit=limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
        ).points
