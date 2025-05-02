import re
import os
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

def analisar_sentimento_transformers(texto, classifier):
    resultado = classifier(texto)
    # Mapeia o índice para o rótulo de sentimento
    mapeamento_sentimentos = {0: "NEGATIVO", 1: "NEUTRO", 2: "POSITIVO"}
    label_idx = resultado[0]['label'].split('_')[1]
    sentimento = mapeamento_sentimentos[int(label_idx)]
    return sentimento, resultado[0]['score']

def analisar_csv_com_transformers(caminho_csv, classifier):
    resultados = []
    with open(caminho_csv, mode='r', encoding='utf-8') as f:
        for linha in f:
            linha = linha.strip()
            # Ignora linhas vazias ou de separador
            if not linha or '----' in linha or linha.startswith('|:'):
                continue
            # Extrai texto e score entre pipes usando regex
            match = re.match(r'^\|\s*(.*?)\s*\|\s*([-\d]+)\s*\|$', linha)
            if match:
                texto = match.group(1)
                label, score = analisar_sentimento_transformers(texto, classifier)
                resultados.append({
                    'texto': texto,
                    'sentimento': label,
                    'score': score
                })
    return resultados

if __name__ == "__main__":
    # Carrega o modelo fine-tunado
    model_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modelo_finetunado_bert_pt'))
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

    caminho_csv = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'teste.csv'))
    resultados = analisar_csv_com_transformers(caminho_csv, classifier)
    for r in resultados:
        print(f"Texto: {r['texto']}\nSentimento: {r['sentimento']} (score: {r['score']:.2f})\n")