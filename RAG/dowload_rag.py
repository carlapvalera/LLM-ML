import torch
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

# Paso 1: Cargar el modelo RAG y el tokenizador
model_name = "facebook/rag-sequence-nq"  # Puedes cambiar esto a "facebook/rag-token-nq" si lo prefieres

# Cargar el tokenizador
tokenizer = RagTokenizer.from_pretrained(model_name)

# Cargar el modelo
model = RagSequenceForGeneration.from_pretrained(model_name)

# Paso 2: Guardar el modelo y el tokenizador localmente
model.save_pretrained("./rag_model")
tokenizer.save_pretrained("./rag_model")

# Paso 3: Configurar el recuperador
retriever = RagRetriever.from_pretrained(model_name, use_dummy_dataset=True)

# Paso 4: Hacer una pregunta
question = "¿Cuál es la capital de Francia?"

# Tokenizar la pregunta
inputs = tokenizer(question, return_tensors="pt")

# Generar respuestas
with torch.no_grad():
    generated_ids = model.generate(input_ids=inputs["input_ids"], 
                                    attention_mask=inputs["attention_mask"],
                                    num_return_sequences=1)

# Decodificar la respuesta generada
generated_answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

# Imprimir la respuesta
print("Pregunta:", question)
print("Respuesta generada:", generated_answer[0])   