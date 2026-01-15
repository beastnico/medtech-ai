import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    BASE_DIR = Path(__file__).resolve().parent

    DATA_DIR = BASE_DIR / "data"
    VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "db_faiss"
    LOGS_DIR = BASE_DIR / "logs"

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    VECTORSTORE_DIR.parent.mkdir(parents=True, exist_ok=True)

    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
    HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set in the environment")

    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    LLM_MODEL = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE = 0.0

    RETRIEVAL_K = 6
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    PAGE_TITLE = "MedTech AI — Biomedical Engineering Assistant"
    PAGE_ICON = "🔬"
    INITIAL_MESSAGE = (
        "Hello! I'm MedTech AI, your biomedical engineering assistant. "
        "Ask me about devices, instrumentation, or biomedical concepts."
    )

    CUSTOM_PROMPT_TEMPLATE = """
You are a biomedical engineering assistant with two modes.

QA mode:
- Answer only using the provided context.
- If the answer is not in the context, say:
  "I don't know based on the available documentation."
- Do not use outside knowledge.
- Be precise and technical.

Chat mode:
- Respond naturally to greetings or casual conversation.

Context:
{context}

User question:
{question}
""".strip()
