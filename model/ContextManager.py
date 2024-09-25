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
        
# Ejemplo de uso de las clases
if __name__ == "__main__":
    # Configurar la base de datos y agregar algunos embeddings (esto se haría solo una vez)
    conn = EmbeddingManager.setup_database()

    questions_to_insert = [
        "¿Cuál es la capital de Francia?",
        "La capital de Francia es París.",
        "¿Quién fue el primer presidente de Estados Unidos?",
        "George Washington fue el primer presidente.",
    ]

    for question in questions_to_insert:
        embedding = EmbeddingManager.generate_embedding(question)
        EmbeddingManager.insert_embedding(conn, embedding, question)  # Guardar también el contexto

    # Cerrar conexión después de insertar los embeddings iniciales
    conn.close()

    # Crear instancia de ContextManager y obtener contexto para una consulta
    context_manager_instance = ContextManager()
    
    query = "¿Cuál es la capital de Francia?"
    
    # Llamar al método para obtener el contexto relevante
    context_result = context_manager_instance._get_context(query)
    
    print("Contexto relevante:", context_result)

    # Guardar una respuesta en la base de datos usando _save_context
    answer_to_save = "La capital de Francia es París."
    context_manager_instance._save_context(answer_to_save)