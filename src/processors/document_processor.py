import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
import inflection
from fastembed import TextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.file_config import FileConfig

logger = logging.getLogger("main_logger")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

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

    def textify_code(self, code: str, context: Dict) -> str:
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

    def process_document(
        self, content: str, metadata: Dict
    ) -> Tuple[List[float], List[float]]:
        """Process a document and return both NLP and code embeddings"""
        # Get text representation
        text_repr = self.textify_code(content, metadata)

        # Generate embeddings
        nlp_embedding = next(self.nlp_model.embed([text_repr]))
        code_embedding = next(self.code_model.embed([content]))

        return nlp_embedding, code_embedding
