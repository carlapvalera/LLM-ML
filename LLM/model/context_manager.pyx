# context_manager.pyx
from EmbeddingManager import EmbeddingManager

cdef class ContextManager:
    def __init__(self):
        pass

    def _get_context(self, query):
        relevant_indices, contexts = EmbeddingManager.relevant_context(query)
        relevant_contexts = [contexts[i] for i in relevant_indices]
        concatenated_context = " ".join(relevant_contexts)
        return concatenated_context

    def _save_context(self, answer):
        embedding = EmbeddingManager.generate_embedding(answer)
        conn = EmbeddingManager.setup_database()
        EmbeddingManager.insert_embedding(conn, embedding, answer)