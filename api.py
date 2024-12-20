import os

from main import ProjectAnalyzer, analyze_project

os.environ["TOKENIZERS_PARALLELISM"] = "true"
PROJECT_ROOT = "."
# # Simple usage
# response = analyze_project(
#     PROJECT_ROOT, "What does the project do?"
# )

# More detailed usage
analyzer = ProjectAnalyzer(str(PROJECT_ROOT))
analyzer.initialize()

# Create chain
chain = analyzer.create_rag_chain()
# Ask questions about the codebase
response = chain.invoke(
    "Based on the actual code implementation, explain what this project does and how it's structured. Reference specific classes and their relationships."
)
print(response)