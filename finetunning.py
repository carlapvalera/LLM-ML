import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Configuración
model_name = "model/saved_model"  # Nombre del modelo en Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Cargar un conjunto de datos (puedes reemplazar esto con tu propio conjunto de datos)
dataset = load_dataset("squad")  # Ejemplo con SQuAD

# Preprocesar los datos
def preprocess_function(examples):
    inputs = examples["question"]
    targets = examples["context"]  # Cambia esto según tu tarea específica
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)

    # Configurar la etiqueta (target)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=512, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Aplicar preprocesamiento al conjunto de datos
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Configuración del entrenamiento
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,  # Cambiado a 10 épocas
    weight_decay=0.01,
    save_total_limit=1,  # Limitar el número de modelos guardados
    save_steps=500,  # Guardar cada 500 pasos
)

# Definir el Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

# Iniciar el fine-tuning
trainer.train()

# Guardar el modelo fine-tuneado en Google Drive (si estás en Google Colab)
model.save_pretrained("/content/drive/MyDrive/fine_tuned_model")
tokenizer.save_pretrained("/content/drive/MyDrive/fine_tuned_model")

print("Fine-tuning completado y modelo guardado en Google Drive.")