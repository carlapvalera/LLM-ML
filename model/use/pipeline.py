import sqlite3
import numpy as np
import torch
from transformers import AutoTokenizer, DPRContextEncoder
from sklearn.metrics.pairwise import cosine_similarity

# Configuración del modelo y tokenizador
save_directory = "./saved_model"
tokenizer = AutoTokenizer.from_pretrained(save_directory)
model = DPRContextEncoder.from_pretrained(save_directory)

# Función para generar embeddings
def generate_embedding(question):
    inputs = tokenizer(question, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).pooler_output
    return embeddings

# Configuración de la base de datos SQLite
def setup_database():
    conn = sqlite3.connect('embeddings.db')
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS embeddings (
        id INTEGER PRIMARY KEY,
        embedding BLOB
    )
    ''')
    conn.commit()
    return conn

# Función para insertar un embedding en la base de datos
def insert_embedding(conn, embedding):
    embedding_blob = embedding.numpy().tobytes()  # Convertir a bytes
    c = conn.cursor()
    c.execute('INSERT INTO embeddings (embedding) VALUES (?)', (embedding_blob,))
    conn.commit()

# Función para recuperar todos los embeddings de la base de datos
def retrieve_all_embeddings(conn):
    c = conn.cursor()
    c.execute('SELECT embedding FROM embeddings')
    rows = c.fetchall()
    
    embeddings_list = []
    
    for row in rows:
        embedding_blob = row[0]
        embedding_array = np.frombuffer(embedding_blob, dtype=np.float32)  # Convertir de bytes a array
        embeddings_list.append(embedding_array)

    return np.array(embeddings_list)

# Función para encontrar los embeddings más relevantes
def relevant_context(query):
    # Generar el embedding para la consulta
    query_embedding = generate_embedding(query).numpy()

    # Configurar la base de datos y recuperar todos los embeddings
    conn = setup_database()
    stored_embeddings = retrieve_all_embeddings(conn)
    
    # Calcular similitudes coseno entre la consulta y los embeddings almacenados
    similarities = cosine_similarity(query_embedding.reshape(1, -1), stored_embeddings).flatten()

    # Obtener índices de los embeddings más similares (por ejemplo, los 3 más cercanos)
    most_relevant_indices = np.argsort(similarities)[-3:][::-1]  # Obtener los 3 índices más altos
    
    print("Consulta:", query)
    print("Índices de los embeddings más relevantes:", most_relevant_indices)
    
    # Cerrar la conexión a la base de datos
    conn.close()
    
    return most_relevant_indices

# Main execution flow
if __name__ == "__main__":
    # Configurar la base de datos
    conn = setup_database()

    # Ejemplo: insertar algunos embeddings en la base de datos (esto se haría una sola vez)
    questions_to_insert = [
        "¿Cuál es la capital de Francia?",
        "¿Quién fue el primer presidente de Estados Unidos?",
        "¿Cuántos planetas hay en el sistema solar?"
    ]

    for question in questions_to_insert:
        embedding = generate_embedding(question)
        insert_embedding(conn, embedding)

    # Cerrar conexión después de insertar los embeddings iniciales
    conn.close()

    # Ejemplo de uso con una consulta
    query = "¿Cuál es la capital de Francia?"
    
    # Llamar a la función para encontrar el contexto relevante
    relevant_indices = relevant_context(query)


    
