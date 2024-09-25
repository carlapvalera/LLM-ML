
from EmbeddingManager import EmbeddingManager
class ContextManager:
    def __init__(self):
        pass
    
    def _get_context(self, query):
        # Obtener índices relevantes y sus contextos usando EmbeddingManager
        relevant_indices, contexts = EmbeddingManager.relevant_context(query)
        
        # Concatenar los contextos relevantes en un solo string
        relevant_contexts = [contexts[i] for i in relevant_indices]
        
        # Unir los contextos en un solo string
        concatenated_context = " ".join(relevant_contexts)
        
        return concatenated_context
    
    def _save_context(self, answer):
        # Generar el embedding para el contexto que se va a guardar
        embedding = EmbeddingManager.generate_embedding(answer)

        # Configurar la base de datos para insertar el nuevo contexto con su embedding
        conn = EmbeddingManager.setup_database()
        
        # Insertar el contexto en la base de datos junto con su embedding correspondiente
        EmbeddingManager.insert_embedding(conn, embedding, answer)
        
