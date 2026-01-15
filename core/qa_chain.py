from typing import Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from utils.logger import setup_logger

logger = setup_logger(__name__)

class QAChainManager:
    def __init__(
        self,
        groq_api_key: str,
        model_name: str = "deepseek-r1-distill-llama-70b",
        temperature: float = 0.0
    ):
        self.groq_api_key = groq_api_key
        self.model_name = model_name
        self.temperature = temperature
        self._llm = None
        self._qa_chain = None
        self._retriever = None

    @property
    def llm(self) -> ChatGroq:
        if self._llm is None:
            logger.info(f"Initializing LLM: {self.model_name}")
            try:
                self._llm = ChatGroq(
                    model_name=self.model_name,
                    temperature=self.temperature,
                    groq_api_key=self.groq_api_key,
                )
                logger.info("LLM initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize LLM: {str(e)}")
                raise
        return self._llm

    def create_qa_chain(
        self,
        retriever,
        prompt_template: str,
        return_source_documents: bool = True
    ):
        try:
            logger.info("Creating QA chain with LCEL")
            self._retriever = retriever
            prompt = ChatPromptTemplate.from_template(prompt_template)
            self._qa_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )
            logger.info("QA chain created successfully")
            return self._qa_chain
        except Exception as e:
            logger.error(f"Failed to create QA chain: {str(e)}")
            raise

    def query(self, question: str) -> Dict[str, Any]:
        if self._qa_chain is None:
            raise ValueError("QA chain not initialized. Call create_qa_chain first.")
        try:
            logger.info(f"Processing query: {question[:50]}...")
            source_docs = []
            if self._retriever:
                try:
                    source_docs = self._retriever.invoke(question)
                except Exception as e:
                    logger.warning(f"Error retrieving documents: {str(e)}")
                    source_docs = []
            else:
                logger.warning("No retriever available for source document retrieval")
            answer = self._qa_chain.invoke(question)
            response = {
                "answer": answer,
                "source_documents": source_docs
            }
            logger.info("Query processed successfully")
            return response
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise

    def format_response(
        self,
        response: Dict[str, Any],
        include_sources: bool = False
    ) -> str:
        result = response.get("answer") or response.get("result", "No answer generated.")
        import re
        result = re.sub(r'浐.*? 浐', '', result, flags=re.DOTALL)
        result = result.strip()
        if not include_sources:
            return result
        source_docs = response.get("source_documents") or response.get("context", [])
        if not source_docs:
            return result
        sources_text = "\n\nSources:\n"
        for i, doc in enumerate(source_docs[:3], 1):
            if hasattr(doc, 'metadata'):
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', 'N/A')
                sources_text += f"{i}. {source} (Page {page})\n"
            else:
                sources_text += f"{i}. Unknown source\n"
        return result + sources_text
