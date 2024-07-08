from flask import Flask, request, jsonify
import torch
from transformers import BertForQuestionAnswering, BertTokenizerFast

app = Flask(__name__)

# Cargar el modelo y el tokenizador
model_path = "./fine_tuned_bert_squad"  # Ruta donde guardaste tu modelo entrenado
model = BertForQuestionAnswering.from_pretrained(model_path)
tokenizer = BertTokenizerFast.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def answer_question(question, context):
    # Tokenizar la pregunta y el contexto
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Obtener predicciones
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Obtener las posiciones de inicio y fin de la respuesta
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits)
    
    # Convertir las posiciones de los tokens a las posiciones de los caracteres en el contexto
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    answer = tokenizer.convert_tokens_to_string(tokens[answer_start:answer_end+1])
    
    # Limpiar la respuesta
    answer = answer.replace("[CLS]", "").replace("[SEP]", "").strip()
    
    return answer

question = "como se llama mi amiguito"
context = "leo es una persomna,Este ejemplo muestra cómo crear una API simple usando Flask para utilizar tu modelo BERT entrenado para responder preguntas. La API acepta solicitudes POST con una pregunta y un contexto, y devuelve la respuesta generada por el modelo. Puedes extender esta API para incluir más funcionalidades según tus necesidades,leo es mi amigo"
answer =answer_question(question, context)
print(answer)

if __name__ == '__main__':
    question = "como se llama mi amiguito"
    context = "leo es una persomna,Este ejemplo muestra cómo crear una API simple usando Flask para utilizar tu modelo BERT entrenado para responder preguntas. La API acepta solicitudes POST con una pregunta y un contexto, y devuelve la respuesta generada por el modelo. Puedes extender esta API para incluir más funcionalidades según tus necesidades,leo es mi amigo"
    print(answer_question(question, context))
