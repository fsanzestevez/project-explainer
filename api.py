import os

from main import ProjectAnalyzer, analyze_project

os.environ["TOKENIZERS_PARALLELISM"] = "true"
PROJECT_ROOT = "."
# Simple usage
response = analyze_project(
    PROJECT_ROOT, "What does the project do?"
)

# More detailed usage
analyzer = ProjectAnalyzer(PROJECT_ROOT)
analyzer.initialize()  # Index the project
chain = analyzer.create_rag_chain()

# Ask questions about the codebase
response = chain.invoke(
    "What does the project do?"
)
print(response)