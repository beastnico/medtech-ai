from pathlib import Path
from typing import Optional
import torch

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from utils.logger import setup_logger

logger = setup_logger(__name__)

try:
    import faiss
    GPU_AVAILABLE = faiss.get_num_gpus() > 0
    logger.info(f"FAISS GPU available: {GPU_AVAILABLE}")
except ImportError:
    GPU_AVAILABLE = False
    logger.info("FAISS GPU not available, using CPU")

class VectorStoreManager:
    
    def __init__(self, embedding_model_name: str, vectorstore_path: Path):
        self.embedding_model_name = embedding_model_name
        self.vectorstore_path = vectorstore_path
        self._embedding_model = None
        self._vectorstore = None
        
    @property
    def embedding_model(self) -> HuggingFaceEmbeddings:
        if self._embedding_model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            logger.info(f"Embedding device: {device}")

            try:
                self._embedding_model = HuggingFaceEmbeddings(
                    model_name=self.embedding_model_name,
                    model_kwargs={"device": device},
                    encode_kwargs={
                        "batch_size": 64
                    }
                )
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {str(e)}")
                raise

        return self._embedding_model
    
    def load_vectorstore(self) -> FAISS:
        if self._vectorstore is not None:
            return self._vectorstore
            
        if not self.vectorstore_path.exists():
            error_msg = f"Vector store not found at {self.vectorstore_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            logger.info(f"Loading vector store from {self.vectorstore_path}")
            try:
                self._vectorstore = FAISS.load_local(
                    str(self.vectorstore_path),
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
            except TypeError:
                self._vectorstore = FAISS.load_local(
                    str(self.vectorstore_path),
                    self.embedding_model
                )

            logger.info("Vector store loaded successfully")
            return self._vectorstore
        except Exception as e:
            logger.error(f"Failed to load vector store: {str(e)}")
            raise
    
    def create_vectorstore(self, documents: list) -> FAISS:
        if not documents:
            raise ValueError("Cannot create vector store from empty document list")
        
        try:
            logger.info(f"Creating vector store with {len(documents)} documents")
            self._vectorstore = FAISS.from_documents(
                documents,
                self.embedding_model
            )
            logger.info("Vector store created successfully")
            return self._vectorstore
        except Exception as e:
            logger.error(f"Failed to create vector store: {str(e)}")
            raise
    
    def save_vectorstore(self) -> None:
        if self._vectorstore is None:
            raise ValueError("No vector store to save. Load or create one first.")
        
        try:
            self.vectorstore_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving vector store to {self.vectorstore_path}")
            self._vectorstore.save_local(str(self.vectorstore_path))
            logger.info("Vector store saved successfully")
        except Exception as e:
            logger.error(f"Failed to save vector store: {str(e)}")
            raise
    
    def get_retriever(self, k: int = 3):
        if self._vectorstore is None:
            self.load_vectorstore()
        
        logger.debug(f"Creating retriever with k={k}")
        return self._vectorstore.as_retriever(search_kwargs={"k": k})
