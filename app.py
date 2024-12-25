from typing import Optional
from fastapi import FastAPI, HTTPException
from src.config.logging_config import setup_logger
from src.analysers.project_analyser import ProjectAnalyzer
import uvicorn


logger = setup_logger("main_logger", "application.log")  # Keep your logger setup

app = FastAPI()


@app.get("/analyze/") # Use a query parameter
async def analyze_endpoint(project_path: str, question: Optional[str]):
    """Analyzes a project given a path and a question."""
    if not question:
        question="What is the purpose of the main class in the codebase?"
    try:
        analyzer = ProjectAnalyzer(project_path)
        analyzer.initialize()
        chain = analyzer.create_rag_chain()
        response = chain.invoke(question)
        return {"response": response}  # Return as JSON
    except Exception as e: # Handle potential errors
        logger.error(f"Error during analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)