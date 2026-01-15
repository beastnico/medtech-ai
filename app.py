import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from core.vectorstore import VectorStoreManager
from core.qa_chain import QAChainManager
from utils.logger import setup_logger

logger = setup_logger(__name__, log_dir=Config.LOGS_DIR)

st.set_page_config(
    page_title=Config.PAGE_TITLE,
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    :root {
        --claude-orange: #D97757;
        --claude-dark: #1F1F1F;
        --claude-light-bg: #F5F5F5;
        --claude-text: #2C2C2C;
        --claude-border: #E5E5E5;
        --claude-hover: #FFF4ED;
    }
    
    .main {
        background-color: #FFFFFF;
    }
    
    .stChatMessage {
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin-bottom: 1rem;
        border: 1px solid var(--claude-border);
    }
    
    [data-testid="stChatMessageContent"] {
        font-size: 1rem;
        line-height: 1.6;
        color: var(--claude-text);
    }
    
    .stChatMessage[data-testid*="user"] {
        background-color: var(--claude-hover);
        border-left: 3px solid var(--claude-orange);
    }
    
    .stChatMessage[data-testid*="assistant"] {
        background-color: #FFFFFF;
        border-left: 3px solid var(--claude-dark);
    }
    
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: var(--claude-dark);
        font-weight: 700;
        font-size: 2.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .stChatInputContainer {
        border-top: 1px solid var(--claude-border);
        padding-top: 1rem;
    }
    
    .stButton > button {
        background-color: var(--claude-orange);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #C86647;
        box-shadow: 0 2px 8px rgba(217, 119, 87, 0.3);
    }
    
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: var(--claude-light-bg);
    }
    
    [data-testid="stMetricValue"] {
        color: var(--claude-orange);
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    hr {
        border-color: var(--claude-border);
        margin: 1.5rem 0;
    }
    
    .stAlert {
        border-radius: 0.5rem;
        border-left: 3px solid;
    }
    
    .stSuccess {
        background-color: #F0F9F4;
        border-left-color: #10A37F;
    }
    
    .stInfo {
        background-color: #F5F5F5;
        border-left-color: var(--claude-orange);
    }
    
    .stError {
        background-color: #FEF2F2;
        border-left-color: #EF4444;
    }
    
    .stSpinner > div {
        border-top-color: var(--claude-orange) !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_system():
    try:
        logger.info("Initializing BME Bot system")
        if not Config.GROQ_API_KEY:
            error_msg = "GROQ_API_KEY not found in environment variables"
            logger.error(error_msg)
            st.error(f"⚠️ Configuration Error: {error_msg}")
            st.stop()
        
        vectorstore_manager = VectorStoreManager(
            embedding_model_name=Config.EMBEDDING_MODEL,
            vectorstore_path=Config.VECTORSTORE_DIR
        )
        
        vectorstore_manager.load_vectorstore()
        retriever = vectorstore_manager.get_retriever(k=Config.RETRIEVAL_K)
        
        qa_manager = QAChainManager(
            groq_api_key=Config.GROQ_API_KEY,
            model_name=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE
        )
        
        qa_manager.create_qa_chain(
            retriever=retriever,
            prompt_template=Config.CUSTOM_PROMPT_TEMPLATE,
            return_source_documents=True
        )
        
        logger.info("System initialization successful")
        return qa_manager, True
        
    except FileNotFoundError as e:
        logger.error(f"Vector store not found: {str(e)}")
        st.error("⚠️ Vector store not found. Please run the indexing script first.")
        st.info("💡 Run: `python scripts/build_vectorstore.py`")
        return None, False
        
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        st.error(f"⚠️ System initialization failed: {str(e)}")
        return None, False

def display_header():
    st.markdown("<h1 class='main-header'>🧬 MedTech AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Intelligent Biomedical Assistant</p>", unsafe_allow_html=True)
    st.divider()

def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        logger.info("New chat session started")
    
    if 'total_queries' not in st.session_state:
        st.session_state.total_queries = 0

def display_chat_history():
    for message in st.session_state.messages:
        avatar = "🧑‍💻" if message['role'] == 'user' else "🤖"
        with st.chat_message(message['role'], avatar=avatar):
            st.markdown(message['content'])

def process_user_query(qa_manager: QAChainManager, user_input: str):
    with st.chat_message('user', avatar="🧑‍💻"):
        st.markdown(user_input)
    
    st.session_state.messages.append({'role': 'user', 'content': user_input})
    st.session_state.total_queries += 1
    
    with st.chat_message('assistant', avatar="🤖"):
        with st.spinner('Processing your question...'):
            try:
                logger.info(f"Processing query #{st.session_state.total_queries}")
                response = qa_manager.query(user_input)
                
                # Determine if this is a question that should include sources
                # Check if it's a greeting or casual conversation
                greeting_keywords = [
                    'hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 
                    'good evening', 'how are you', 'howdy', 'hey there', 'hi there',
                    'what\'s up', 'how\'s it going', 'nice to meet you', 'pleased to meet you'
                ]
                
                # Check if the input is purely a greeting (no question marks or technical terms)
                user_lower = user_input.lower().strip()
                is_pure_greeting = any(
                    keyword == user_lower or 
                    (keyword in user_lower and '?' not in user_input and 
                     not any(tech_word in user_lower for tech_word in ['calibrate', 'machine', 'dialysis', 'how do', 'what is', 'explain', 'describe', 'calculate', 'procedure', 'method', 'process']))
                    for keyword in greeting_keywords
                )
                
                # Only include sources for actual questions, not greetings
                include_sources = not is_pure_greeting
                
                result = qa_manager.format_response(
                    response,
                    include_sources=include_sources
                )
                st.markdown(result)
                st.session_state.messages.append({'role': 'assistant', 'content': result})
                logger.info("Query processed successfully")
            except Exception as e:
                error_msg = f"I encountered an error: {str(e)}"
                logger.error(f"Query processing error: {str(e)}")
                st.error(error_msg)
                st.session_state.messages.append({'role': 'assistant', 'content': error_msg})

def display_sidebar(qa_manager):
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **MedTech AI** is an AI-powered assistant specialized in biomedical engineering topics.
        
        **Features:**
        - 🔍 RAG-based answers from technical documents
        - 💬 Natural conversation support
        - 🎯 Context-aware responses
        - ⚡ Powered by advanced language models
        
        **Tech Stack:**
        - LangChain + FAISS
        - Groq LLM (DeepSeek)
        - Streamlit Interface
        """)
        
        st.divider()
        
        st.header("📊 Session Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Queries", st.session_state.total_queries)
        with col2:
            st.metric("Messages", len(st.session_state.messages))
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.total_queries = 0
            logger.info("Chat history cleared")
            st.rerun()
        
        st.divider()
        
        st.header("⚙️ System Status")
        st.success("✓ Vector Store Loaded")
        st.success("✓ LLM Connected")
        st.info(f"🤖 Model: {Config.LLM_MODEL}")

def main():
    display_header()
    initialize_session_state()
    qa_manager, initialized = initialize_system()
    
    if not initialized:
        st.stop()
    
    display_sidebar(qa_manager)
    display_chat_history()
    
    if len(st.session_state.messages) == 0:
        with st.chat_message('assistant', avatar="🤖"):
            st.markdown(Config.INITIAL_MESSAGE)
        st.session_state.messages.append({
            'role': 'assistant',
            'content': Config.INITIAL_MESSAGE
        })

    if user_input := st.chat_input("Ask me anything about biomedical engineering..."):
        process_user_query(qa_manager, user_input)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Application crashed: {str(e)}", exc_info=True)
        st.error("🚨 Critical error occurred. Please check logs.")
        st.exception(e)
