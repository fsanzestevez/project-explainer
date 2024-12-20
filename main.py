import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import inflection
from fastembed import TextEmbedding
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (NotebookLoader, TextLoader,
                                                  UnstructuredMarkdownLoader)
from langchain_ollama import OllamaLLM
from pathspec import PathSpec
from qdrant_client import QdrantClient, models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FileConfig:
    """Configuration for handling different file types"""

    extensions: List[str]
    chunk_size: int
    chunk_overlap: int
    separators: Optional[List[str]] = None


def textify_code(code: str, context: Dict) -> str:
    """Convert code to natural language representation"""
    # Get module and filename from context
    module = context.get("module", "")
    filename = context.get("file_name", "")

    # Extract function/class names using simple regex
    names = re.findall(r"(?:class|def)\s+(\w+)", code)
    name = names[0] if names else ""

    # Convert to human readable
    if name:
        name = inflection.humanize(inflection.underscore(name))

    # Build text representation
    text = f"{name} in module {module} file {filename} "

    # Add code content without special characters
    code_text = " ".join(re.split(r"\W+", code))
    text += code_text

    return text.strip()


class DocumentProcessor:
    """document processor with dual embedding support"""

    FILE_CONFIGS = {
        "python": FileConfig(
            extensions=["py"],
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\nclass ", "\ndef ", "\n\n"],
        ),
        "notebook": FileConfig(
            extensions=["ipynb"], chunk_size=1000, chunk_overlap=200
        ),
        "docker": FileConfig(
            extensions=["dockerfile"], chunk_size=500, chunk_overlap=50
        ),
        "config": FileConfig(
            extensions=["yaml", "yml", "json"], chunk_size=500, chunk_overlap=50
        ),
        "markdown": FileConfig(
            extensions=["md", "rst"], chunk_size=1000, chunk_overlap=200
        ),
    }

    def __init__(self):
        self.file_type_map = self._build_extension_map()
        self.nlp_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
        self.code_model = TextEmbedding("jinaai/jina-embeddings-v2-base-code")

    def _build_extension_map(self) -> Dict[str, str]:
        """Creates mapping from file extensions to file types"""
        ext_map = {}
        for file_type, config in self.FILE_CONFIGS.items():
            for ext in config.extensions:
                ext_map[ext] = file_type
        return ext_map

    def get_file_type(self, file_path: str) -> Optional[str]:
        """Determines file type from extension"""
        ext = Path(file_path).suffix.lower().lstrip(".")
        return self.file_type_map.get(ext)

    def create_splitter(self, file_type: str) -> RecursiveCharacterTextSplitter:
        """Creates appropriate splitter for file type"""
        config = self.FILE_CONFIGS[file_type]
        return RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators or ["\n\n", "\n", " ", ""],
        )

    def process_document(
        self, content: str, metadata: Dict
    ) -> Tuple[List[float], List[float]]:
        """Process a document and return both NLP and code embeddings"""
        # Get text representation
        text_repr = textify_code(content, metadata)

        # Generate embeddings
        nlp_embedding = next(self.nlp_model.embed([text_repr]))
        code_embedding = next(self.code_model.embed([content]))

        return nlp_embedding, code_embedding


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
                logger.error(f"Error processing {file_path}: {str(e)}")

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


class ProjectAnalyzer:
    """main interface for analyzing codebases"""

    def __init__(self, project_path: str):
        self.indexer = CodebaseIndexer(project_path)
        self.llm = OllamaLLM(model="codellama", temperature=0.1, num_ctx=8192)

    def initialize(self):
        """Indexes the project and prepares the system"""
        logger.info("Indexing project files...")
        self.indexer.index_project()
        logger.info("Project indexing complete")

    def create_rag_chain(self):
        """Creates the RAG chain with  context handling"""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are analyzing specific code from a codebase. Focus ONLY on the provided code context.
                Analyze:
                1. Implementation details shown in the code
                2. Specific classes, functions, and their relationships
                3. Actual patterns and structures present in the code
                4. Technical details that are explicitly present
                
                DO NOT make assumptions about functionality not shown in the code.
                Always reference specific code elements in your explanation.""",
                ),
                (
                    "user",
                    "Code context:\n\n{context}\n\nBased on the provided code, {question}",
                ),
            ]
        )

        def format_context(results):
            return "\n\n".join(
                [
                    f"File: {r.payload['metadata']['file_path']}\n"
                    f"Content:\n{r.payload['content']}"
                    for r in results
                ]
            )

        chain = (
            {
                "context": lambda x: format_context(self.indexer.search(x)),
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain


def analyze_project(project_path: str, question: str) -> str:
    """Helper function to analyze a project"""
    analyzer = ProjectAnalyzer(project_path)
    analyzer.initialize()
    chain = analyzer.create_rag_chain()
    return chain.invoke(question)
