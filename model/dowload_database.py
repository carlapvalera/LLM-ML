from datasets import load_dataset

# Cargar el conjunto de datos completo
dataset = load_dataset("shamanez/RAG-end2end", split="train+validation")  # Cargar ambos splits

# Guardar el conjunto de datos en un directorio local
dataset.save_to_disk("./rag_dataset")