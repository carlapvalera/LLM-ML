import os
from transformers import AutoTokenizer, DPRContextEncoder

# Define el nombre del modelo
model_name = "Oscar066/RAG-end2end-Model"

# Crea un directorio para guardar el modelo si no existe
save_directory = "./saved_model"
os.makedirs(save_directory, exist_ok=True)

# Cargar el tokenizador y el modelo
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = DPRContextEncoder.from_pretrained(model_name)

# Guardar el tokenizador y el modelo en el directorio especificado
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

print(f"Modelo y tokenizador guardados en: {save_directory}")