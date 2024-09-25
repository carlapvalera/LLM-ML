import sqlite3
import numpy as np
import torch
from transformers import AutoTokenizer, DPRContextEncoder
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingManager:
    save_directory = "C:\\blabla\\LLM-ML\\saved_model"
    tokenizer = AutoTokenizer.from_pretrained(save_directory)
    model = DPRContextEncoder.from_pretrained(save_directory)

    @staticmethod
    def setup_database():
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

    @staticmethod
    def generate_embedding(query):
        inputs = EmbeddingManager.tokenizer(query, return_tensors="pt")
        with torch.no_grad():
            embeddings = EmbeddingManager.model(**inputs).pooler_output
        return embeddings

    @staticmethod
    def insert_embedding(conn, embedding, context):
        embedding_blob = embedding.numpy().tobytes()  # Convertir a bytes
        c = conn.cursor()
        c.execute('INSERT INTO embeddings (embedding, context) VALUES (?, ?)', (embedding_blob, context))
        conn.commit()

    @staticmethod
    def retrieve_all_embeddings(conn):
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

    @staticmethod
    def relevant_context(query):
        # Generar el embedding para la consulta
        query_embedding = EmbeddingManager.generate_embedding(query).numpy()

        # Configurar la base de datos y recuperar todos los embeddings
        conn = EmbeddingManager.setup_database()
        stored_embeddings, contexts = EmbeddingManager.retrieve_all_embeddings(conn)
        
        # Calcular similitudes coseno entre la consulta y los embeddings almacenados
        similarities = cosine_similarity(query_embedding.reshape(1, -1), stored_embeddings).flatten()

        # Obtener índices de los embeddings más similares (por ejemplo, los 3 más cercanos)
        most_relevant_indices = np.argsort(similarities)[-3:][::-1]  # Obtener los 3 índices más altos
        
        # Cerrar la conexión a la base de datos
        conn.close()
        
        return most_relevant_indices, contexts

