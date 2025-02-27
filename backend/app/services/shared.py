from ..core import logger

# Global RAG engine instance
_rag_engine = None

def get_rag_engine():
    """Get or initialize the global RAG engine instance"""
    global _rag_engine
    from .rag_engine import EnhancedRAGEngine
    
    if _rag_engine is None:
        logger.info("Initializing global RAG engine")
        _rag_engine = EnhancedRAGEngine()
        logger.info(f"RAG engine initialized with ID: {id(_rag_engine)}")
        logger.info(f"RAG engine has {len(_rag_engine.chunks)} chunks loaded")
    else:
        logger.info(f"Returning existing RAG engine with ID: {id(_rag_engine)}")
    
    return _rag_engine