import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from core.document_processor import DocumentProcessor
from core.vectorstore import VectorStoreManager
from utils.logger import setup_logger

logger = setup_logger(__name__, log_dir=Config.LOGS_DIR)

def main():
    print("=" * 60)
    print("MedTech AI - Vector Store Builder")
    print("=" * 60)
    
    try:
        # Validate inputs
        if not Config.DATA_DIR.exists():
            msg = f"Data directory missing at: {Config.DATA_DIR}"
            logger.error(msg)
            print(f"Error: {msg}")
            print("Please create the directory and add your PDF files.")
            return
        
        print(f"\nSource: {Config.DATA_DIR}")
        
        # 1. Process & Chunk
        print("Reading and chunking documents...")
        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
        chunks = doc_processor.process_directory(Config.DATA_DIR)
        
        if not chunks:
            logger.warning("No text extracted. Directory might be empty or PDFs unreadable.")
            print("No documents processed. Check your data folder.")
            return
        
        print(f"Success: Generated {len(chunks)} text chunks.")

        # 2. Embed & Save
        print("Generating embeddings and building index...")
        vectorstore_manager = VectorStoreManager(
            embedding_model_name=Config.EMBEDDING_MODEL,
            vectorstore_path=Config.VECTORSTORE_DIR
        )

        vectorstore_manager.create_vectorstore(chunks)
        vectorstore_manager.save_vectorstore()

        # Final Summary
        print("\n" + "=" * 60)
        print("Vector store built successfully!")
        print("=" * 60)
        print(f"  - Total Chunks:  {len(chunks)}")
        print(f"  - Model Used:    {Config.EMBEDDING_MODEL}")
        print(f"  - Output Path:   {Config.VECTORSTORE_DIR}")
        print(f"\nNext: Run 'streamlit run app.py'")

    except Exception as e:
        logger.exception("Vector store build failed")
        print(f"\nCRITICAL ERROR: {str(e)}")
        print("Check logs for full traceback.")
        sys.exit(1)

if __name__ == "__main__":
    main()