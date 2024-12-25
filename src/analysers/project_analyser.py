import logging

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_ollama import OllamaLLM
from langchain_google_vertexai import ChatVertexAI

from src.indexers.codebase_indexer import CodebaseIndexer

logger = logging.getLogger("main_logger")


class ProjectAnalyzer:
    """main interface for analyzing codebases"""

    def __init__(self, project_path: str):
        self.indexer = CodebaseIndexer(project_path)
        # self.llm = OllamaLLM(model="codellama", temperature=0.1, num_ctx=8192)
        self.llm = ChatVertexAI(
            model="gemini-1.5-flash-001",
            project="gen-lang-client-0979558974",
            temperature=0,
            max_tokens=None,
            max_retries=6,
            stop=None,
            # other params...
        )

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

                Use only provided context for your assumptions and explanations.
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

# curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=AIzaSyA7MMSuBEuOObDZtjhHpF4UuvDMOFPALB0" \
# -H 'Content-Type: application/json' \
# -X POST \
# -d '{
#   "contents": [{
#     "parts":[{"text": "Explain how AI works"}]
#     }]
#    }'