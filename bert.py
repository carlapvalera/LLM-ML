import torch
from torch.utils.data import DataLoader
from transformers import BertForQuestionAnswering, BertTokenizerFast, AdamW
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import TensorDataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# 1. Cargar el dataset SQuAD
print("Cargando el dataset...")
dataset = load_dataset("squad")

# Recortar el dataset para pruebas rápidas
train_size = 8000  # Número de ejemplos de entrenamiento
eval_size = 2000    # Número de ejemplos de evaluación

dataset["train"] = dataset["train"].select(range(train_size))
dataset["validation"] = dataset["validation"].select(range(eval_size))

# 2. Cargar el modelo BERT y el tokenizador
print("Cargando el modelo BERT y el tokenizador...")
model_name = "bert-base-uncased"
model = BertForQuestionAnswering.from_pretrained(model_name)
tokenizer = BertTokenizerFast.from_pretrained(model_name)

# 3. Preprocesar los datos
print("Preprocesando los datos...")
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

# 4. Preparar los dataloaders
print("Preparando los dataloaders...")
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]

# Convertir los datasets a TensorDatasets
def convert_to_tensordataset(dataset):
    print("Convertir los datasets a TensorDatasets")
    return TensorDataset(
        torch.tensor(dataset['input_ids']),
        torch.tensor(dataset['attention_mask']),
        torch.tensor(dataset['start_positions']),
        torch.tensor(dataset['end_positions'])
    )

train_dataset = convert_to_tensordataset(train_dataset)
eval_dataset = convert_to_tensordataset(eval_dataset)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(eval_dataset, batch_size=8)

# 5. Configurar el entrenamiento
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

# 6. Función de entrenamiento
def process_batch(batch, device):
    if isinstance(batch, list):
        # Si es una lista, asumimos que los elementos están en este orden
        input_ids, attention_mask, start_positions, end_positions = batch
        return {
            'input_ids': input_ids.to(device),
            'attention_mask': attention_mask.to(device),
            'start_positions': start_positions.to(device),
            'end_positions': end_positions.to(device)
        }
    elif isinstance(batch, dict):
        return {k: v.to(device) for k, v in batch.items()}
    else:
        raise ValueError(f"Unexpected batch type: {type(batch)}")

def train(model, dataloader, optimizer, device):
    model.train()
    for batch in tqdm(dataloader, desc="Entrenando"):
        batch = process_batch(batch, device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# 7. Función de evaluación
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluando"):
            batch = process_batch(batch, device)
            outputs = model(**batch)
            total_loss += outputs.loss.item()

            # Obtener predicciones
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            start_pred = torch.argmax(start_logits, dim=1)
            end_pred = torch.argmax(end_logits, dim=1)

            # Aplanar las predicciones y las etiquetas
            predictions = torch.stack((start_pred, end_pred), dim=1).view(-1).cpu().numpy()
            labels = torch.stack((batch['start_positions'], batch['end_positions']), dim=1).view(-1).cpu().numpy()

            all_predictions.extend(predictions)
            all_labels.extend(labels)

    # Calcular métricas
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='macro')

    return total_loss / len(dataloader), accuracy, f1

# 8. Entrenamiento y evaluación
num_epochs = 3
print(f"Comenzando entrenamiento por {num_epochs} épocas...")
for epoch in range(num_epochs):
    print(f"Época {epoch + 1}/{num_epochs}")
    train(model, train_dataloader, optimizer, device)
    loss, accuracy, f1 = evaluate(model, eval_dataloader, device)
    print(f"Pérdida de validación: {loss:.4f}")
    print(f"Exactitud de validación: {accuracy:.2%}")
    print(f"F1-score de validación: {f1:.4f}")

# 9. Guardar el modelo
print("Guardando el modelo...")
model.save_pretrained("./fine_tuned_bert_squad")
tokenizer.save_pretrained("./fine_tuned_bert_squad")

print("¡Entrenamiento completado!")