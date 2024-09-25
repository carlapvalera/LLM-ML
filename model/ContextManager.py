
import sqlite3
import numpy as np
import torch
from transformers import AutoTokenizer, DPRContextEncoder
from sklearn.metrics.pairwise import cosine_similarity
#from EmbeddingManager import EmbeddingManager


import sqlite3
import numpy as np
import torch
from transformers import AutoTokenizer, DPRContextEncoder
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingManager:
    def __init__(self, save_directory="C:\\blabla\\LLM-ML\\saved_model"):
        self.save_directory = save_directory
        self.tokenizer = AutoTokenizer.from_pretrained(self.save_directory)
        self.model = DPRContextEncoder.from_pretrained(self.save_directory)

    def setup_database(self):
        conn = sqlite3.connect('embeddings.db')
        c = conn.cursor()
        c.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY,
            embedding BLOB,
            context TEXT
        )
        ''')
        conn.commit()
        return conn

    def generate_embedding(self, query):
        inputs = self.tokenizer(query, return_tensors="pt")
        with torch.no_grad():
            embeddings = self.model(**inputs).pooler_output
        return embeddings

    def insert_embedding(self, conn, embedding, context):
        embedding_blob = embedding.numpy().tobytes()  # Convertir a bytes
        c = conn.cursor()
        c.execute('INSERT INTO embeddings (embedding, context) VALUES (?, ?)', (embedding_blob, context))
        conn.commit()

    def retrieve_all_embeddings(self, conn):
        c = conn.cursor()
        c.execute('SELECT embedding, context FROM embeddings')
        rows = c.fetchall()
        
        embeddings_list = []
        contexts_list = []
        
        for row in rows:
            embedding_blob = row[0]
            context = row[1]
            embedding_array = np.frombuffer(embedding_blob, dtype=np.float32)  # Convertir de bytes a array
            embeddings_list.append(embedding_array)
            contexts_list.append(context)

        return np.array(embeddings_list), contexts_list

    def relevant_context(self, query):
        # Generar el embedding para la consulta
        query_embedding = self.generate_embedding(query).numpy()

        # Configurar la base de datos y recuperar todos los embeddings
        conn = self.setup_database()
        stored_embeddings, contexts = self.retrieve_all_embeddings(conn)
        
        # Calcular similitudes coseno entre la consulta y los embeddings almacenados
        similarities = cosine_similarity(query_embedding.reshape(1, -1), stored_embeddings).flatten()

        # Obtener índices de los embeddings más similares (por ejemplo, los 3 más cercanos)
        most_relevant_indices = np.argsort(similarities)[-3:][::-1]  # Obtener los 3 índices más altos
        
        # Cerrar la conexión a la base de datos
        conn.close()
        
        return most_relevant_indices, contexts

# Ejemplo de uso
if __name__ == "__main__":
    manager = EmbeddingManager()  # Crear una instancia de EmbeddingManager
    query = "¿Cuál es la capital de Francia?"
    
    # Usar el método relevante_context en la instancia
    indices, contexts = manager.relevant_context(query)
    
    print("Índices relevantes:", indices)
    print("Contextos relevantes:", contexts)


class ContextManager:
    def __init__(self, manager):
        self.manager = manager
       
    
    def _get_context(self, query):
        # Obtener índices relevantes y sus contextos usando EmbeddingManager
        relevant_indices, contexts = self.manager.relevant_context(query)
        
        # Concatenar los contextos relevantes en un solo string
        relevant_contexts = [contexts[i] for i in relevant_indices]
        
        # Unir los contextos en un solo string
        concatenated_context = " ".join(relevant_contexts)
        
        return concatenated_context
    
    def _save_context(self, answer):
        # Generar el embedding para el contexto que se va a guardar
        embedding = self.manager.generate_embedding(answer)

        # Configurar la base de datos para insertar el nuevo contexto con su embedding
        conn = self.manager.setup_database()
        
        # Insertar el contexto en la base de datos junto con su embedding correspondiente
        self.manager.insert_embedding(conn, embedding, answer)
        
# Ejemplo de uso de las clases
if __name__ == "__main__":
    # Configurar la base de datos y agregar algunos embeddings (esto se haría solo una vez)
    # Crear una instancia de EmbeddingManager
    manager = EmbeddingManager()

    # Llamar al método setup_database en la instancia
    conn = manager.setup_database()

    questions_to_insert = [
        "¿Cuál es la capital de Francia?",
        "La capital de Francia es París.",
        "¿Quién fue el primer presidente de Estados Unidos?",
        "George Washington fue el primer presidente.",
    ]

    for question in questions_to_insert:
        embedding = manager.generate_embedding(question)
        manager.insert_embedding(conn, embedding, question)  # Guardar también el contexto

    # Cerrar conexión después de insertar los embeddings iniciales
    conn.close()

    # Crear instancia de ContextManager y obtener contexto para una consulta
    context_manager_instance = ContextManager(manager)
    
    query = "¿Cuál es la capital de Francia?"
    
    # Llamar al método para obtener el contexto relevante
    context_result = context_manager_instance._get_context(query)
    
    print("Contexto relevante:", context_result)

    # Guardar una respuesta en la base de datos usando _save_context
    answer_to_save = "La capital de Francia es París."
    context_manager_instance._save_context(answer_to_save)
