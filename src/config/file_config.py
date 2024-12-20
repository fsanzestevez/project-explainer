from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FileConfig:
    """Configuration for handling different file types"""

    extensions: List[str]
    chunk_size: int
    chunk_overlap: int
    separators: Optional[List[str]] = None
