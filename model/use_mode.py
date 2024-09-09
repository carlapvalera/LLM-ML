
from transformers import AutoTokenizer, DPRContextEncoder
import torch
# Cargar el modelo y el tokenizador desde el directorio guardado
save_directory = "./saved_model" 
tokenizer = AutoTokenizer.from_pretrained(save_directory)
model = DPRContextEncoder.from_pretrained(save_directory)

# Ahora puedes usar el modelo y el tokenizador como antes
question = "¿Cuál es la capital de Francia?"
inputs = tokenizer(question, return_tensors="pt")

# Generar representaciones
with torch.no_grad():
    embeddings = model(**inputs).pooler_output

print("Embeddings:", embeddings)