from main import analyze_project, ProjectAnalyzer
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
PROJECT_ROOT = "."
# Simple usage
response = analyze_project(PROJECT_ROOT, "How do the Docker configurations interact with the Python scripts?")

# More detailed usage
analyzer = ProjectAnalyzer(PROJECT_ROOT)
analyzer.initialize()  # Index the project
chain = analyzer.create_rag_chain()

# Ask questions about the codebase
response = chain.invoke("Explain the project's configuration management system")