from src.config.logging_config import setup_logger

logger = setup_logger("main_logger", "application.log")
from src.analysers.project_analyser import ProjectAnalyzer


def analyze_project(project_path: str, question: str) -> str:
    """Helper function to analyze a project"""
    analyzer = ProjectAnalyzer(project_path)
    analyzer.initialize()
    chain = analyzer.create_rag_chain()
    return chain.invoke(question)


if __name__ == "__main__":
    # Example usage
    project_path = "/path/to/project"
    question = "What is the purpose of the main class in the codebase?"
    response = analyze_project(project_path, question)
    print(response)
