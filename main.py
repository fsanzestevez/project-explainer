import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import yaml
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import (NotebookLoader, TextLoader,
                                                  UnstructuredMarkdownLoader)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from pathspec import PathSpec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FileConfig:
    """Configuration for handling different file types"""

    extensions: List[str]
    chunk_size: int
    chunk_overlap: int
    separators: Optional[List[str]] = None


class DocumentProcessor:
    """Handles document loading and splitting for different file types"""

    FILE_CONFIGS = {
        "python": FileConfig(extensions=["py"], chunk_size=1000, chunk_overlap=200),
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
            chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
        )

    def load_file(self, file_path: str) -> List[Document]:
        """Loads and processes a single file"""
        file_type = self.get_file_type(file_path)
        if not file_type:
            logger.warning(f"Unsupported file type: {file_path}")
            return []

        try:
            if file_type == "notebook":
                loader = NotebookLoader(file_path, include_outputs=True)
                docs = loader.load()
            elif file_type == "config":
                docs = self._load_config_file(file_path)
            elif file_type == "markdown":
                # Fallback to TextLoader if UnstructuredMarkdownLoader fails
                try:
                    loader = UnstructuredMarkdownLoader(file_path)
                    docs = loader.load()
                except ImportError:
                    loader = TextLoader(file_path)
                    docs = loader.load()
            else:
                loader = TextLoader(file_path)
                docs = loader.load()

            splitter = self.create_splitter(file_type)
            split_docs = splitter.split_documents(docs)

            # Add metadata
            for doc in split_docs:
                doc.metadata.update(
                    {
                        "file_path": file_path,
                        "file_type": file_type,
                        "project_root": str(Path(file_path).parent),
                    }
                )

            return split_docs

        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return []

    def _load_config_file(self, file_path: str) -> List[Document]:
        """Handles loading of configuration files"""
        with open(file_path, "r") as f:
            ext = Path(file_path).suffix.lower()
            if ext in [".yaml", ".yml"]:
                content = yaml.safe_load(f)
                text = yaml.dump(content, default_flow_style=False)
            else:  # JSON
                content = json.load(f)
                text = json.dumps(content, indent=2)

        return [Document(page_content=text)]


class CodebaseIndexer:
    """Manages vector indices for the codebase"""

    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.processor = DocumentProcessor()
        self.embeddings = HuggingFaceEmbeddings(
            model_name="hkunlp/instructor-xl"
        )
        self.indices: Dict[str, Chroma] = {}
        self.gitignore_spec = self._load_gitignore()

    def _load_gitignore(self) -> Optional[PathSpec]:
        """Loads and parses the .gitignore file if it exists"""
        gitignore_path = self.project_path / ".gitignore"
        if gitignore_path.is_file():
            with gitignore_path.open("r") as f:
                patterns = f.readlines()
            return PathSpec.from_lines("gitwildmatch", patterns)
        return None

    def index_project(self):
        """Indexes all supported files in the project"""
        for file_path in self._get_project_files():
            docs = self.processor.load_file(str(file_path))
            if docs:
                file_type = docs[0].metadata["file_type"]
                if file_type not in self.indices:
                    self.indices[file_type] = Chroma(
                        collection_name=f"codebase_{file_type}",
                        embedding_function=self.embeddings,
                    )
                self.indices[file_type].add_documents(docs)
                logger.info(f"Indexed {len(docs)} documents from {file_path}")
        for file_type, index in self.indices.items():
            num_docs = (
                index._collection.count()
            )  # Use the count() method of the Chroma collection
            logger.info(f"Indexed {num_docs} documents for file type {file_type}")
        logger.info("Project indexing completed")

    def _get_project_files(self) -> Iterator[Path]:
        """Yields all supported files in the project, skipping those in .gitignore"""
        for path in self.project_path.rglob("*"):
            # Skip files and directories listed in .gitignore
            if self.gitignore_spec and self.gitignore_spec.match_file(
                str(path.relative_to(self.project_path))
            ):
                continue
            if path.is_file() and self.processor.get_file_type(str(path)):
                yield path

    def get_combined_retriever(self, k: int = 4):
        """Creates a retriever that searches across all indices"""
        retrievers = []
        for index in self.indices.values():
            retrievers.append(
                index.as_retriever(
                    search_kwargs={"k": min(k, index.get().get("ids", []).__len__())}
                )
            )

        # Use the first retriever if available
        return retrievers[0] if retrievers else None


class ProjectAnalyzer:
    """Main interface for analyzing codebases"""

    def __init__(self, project_path: str):
        self.indexer = CodebaseIndexer(project_path)
        self.llm = OllamaLLM(model="codellama")

    def initialize(self):
        """Indexes the project and prepares the RAG system"""
        logger.info("Indexing project files...")
        self.indexer.index_project()
        logger.info("Project indexing complete")

    def create_rag_chain(self):
        """Creates the RAG chain for answering queries"""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a code analysis expert. Based on the provided context
            from different project files, explain how the components work together.
            Focus on:
            1. Implementation details and code structure
            2. Relationships between configurations and code
            3. Project architecture and design patterns
            4. Dependencies and interactions between components
            
            Provide concrete examples from the context when relevant.""",
                ),
                ("human", "{question}"),
            ]
        )

        retriever = self.indexer.get_combined_retriever()
        if not retriever:
            raise ValueError("No indices available. Run initialize() first.")

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
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
