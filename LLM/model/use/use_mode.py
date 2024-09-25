import sqlite3
import numpy as np
import torch
from transformers import AutoTokenizer, DPRContextEncoder

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
        question TEXT,
        embedding BLOB
    )
    ''')
    conn.commit()
    return conn

# Función para insertar un embedding en la base de datos
def insert_embedding(conn, question, embedding):
    embedding_blob = embedding.numpy().tobytes()  # Convertir a bytes
    c = conn.cursor()
    c.execute('INSERT INTO embeddings (question, embedding) VALUES (?, ?)', (question, embedding_blob))
    conn.commit()

# Función para recuperar y mostrar los embeddings de la base de datos
def retrieve_embeddings(conn):
    c = conn.cursor()
    c.execute('SELECT question, embedding FROM embeddings')
    rows = c.fetchall()
    
    for row in rows:
        question, embedding_blob = row
        embedding_array = np.frombuffer(embedding_blob, dtype=np.float32)  # Convertir de bytes a array
        print("Pregunta:", question)
        print("Embedding:", embedding_array)

# Main execution flow
if __name__ == "__main__":
    # Configurar la base de datos
    conn = setup_database()

    # Pregunta para generar el embedding
    question = "¿Cuál es la capital de Francia?"
    
    # Generar el embedding
    embeddings = generate_embedding(question)

    # Insertar el embedding en la base de datos
    insert_embedding(conn, question, embeddings)

    # Recuperar y mostrar los embeddings almacenados
    print("\nEmbeddings almacenados en la base de datos:")
    retrieve_embeddings(conn)

    # Cerrar la conexión a la base de datos
    conn.close()
    