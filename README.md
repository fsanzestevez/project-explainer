# Project-Explainer

## Project Overview

This project provides a framework for analyzing codebases using dual embeddings and retrieval-augmented generation (RAG). The system is designed to:

1. Process various file types and generate embeddings for both natural language (NLP) and code representations.
2. Index codebases into a vector database (Qdrant) for efficient retrieval.
3. Enable advanced code analysis through a chain powered by an LLM and context-sensitive prompts.

---

## Key Components

### 1. FileConfig
Defines configuration parameters for handling different file types, including:
- Supported extensions
- Chunk sizes
- Chunk overlap
- Optional separators

### 2. DocumentProcessor
Handles file processing, including:
- Determining file types based on extensions
- Splitting files into chunks
- Generating embeddings for both NLP and code representations using pre-trained models.

### 3. CodebaseIndexer
Indexes project files into Qdrant using dual embeddings:
- Utilizes `.gitignore` for excluding files.
- Processes files in batches for efficiency.
- Supports combined search using text and code embeddings.

### 4. ProjectAnalyzer
Main interface for analyzing codebases:
- Initializes the indexing process.
- Creates a RAG chain for context-aware question-answering using the indexed data.

---

## Features

### File Type Support
The system supports a variety of file types, including:
- Python scripts (`.py`)
- Jupyter Notebooks (`.ipynb`)
- Dockerfiles (`dockerfile`)
- Configuration files (`.yaml`, `.yml`, `.json`)
- Markdown files (`.md`, `.rst`)

### Dual Embedding Support
Two pre-trained embedding models are used:
- **NLP Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Code Model:** `jinaai/jina-embeddings-v2-base-code`

### Vector Database
The system uses Qdrant for:
- Storing embeddings
- Enabling efficient similarity search
- Combining text and code search results using fusion techniques

### Retrieval-Augmented Generation (RAG)
A custom RAG chain is built for:
- Contextualizing user queries with relevant code snippets.
- Generating precise and detailed responses with an LLM (`codellama`).

---

## Setup

### Requirements
- Python 3.8+
- Dependencies listed in `requirements.txt`

### Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Initialize the Analyzer
Run the following code to initialize the system:
```python
from <module_name> import ProjectAnalyzer

analyzer = ProjectAnalyzer("<project_path>")
analyzer.initialize()
```

### 2. Perform Analysis
Use the RAG chain to answer questions about the codebase:
```python
chain = analyzer.create_rag_chain()
response = chain.invoke("<your_question>")
print(response)
```

### 3. Example
```python
from <module_name> import analyze_project

project_path = "/path/to/project"
question = "What is the purpose of the main class in the codebase?"
response = analyze_project(project_path, question)
print(response)
```

---

## Logging
Logging is configured at the `INFO` level by default. You can adjust the logging level as needed by modifying the `logging.basicConfig` settings.

---

## Future Enhancements
- Expand support for additional file types.
- Optimize embeddings for larger datasets.
- Improve search accuracy using advanced ranking algorithms.

---

## Contributing
Contributions are welcome! Please submit issues or pull requests to help improve the project.

---

## License
This project is licensed under the [MIT License](LICENSE).

