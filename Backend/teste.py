from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import os

# Caminho absoluto e normalizado para o modelo
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.normpath(os.path.join(base_dir, 'modelo_finetunado_bert_pt'))

model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
clf = pipeline("text-classification", model=model, tokenizer=tokenizer)

print(clf("Desmatamento no Cerrado supera Amazônia e bate recorde em 2024"))