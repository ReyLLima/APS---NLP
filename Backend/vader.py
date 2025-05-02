import csv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os


def carregar_custom_words(caminho_csv):
    custom_dict = {}
    with open(caminho_csv, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            palavra = row['palavra'].strip()
            try:
                score = float(row['score'])
                custom_dict[palavra] = score
            except ValueError:
                continue
    return custom_dict


def analisar_com_vader(texto):
    analyzer = SentimentIntensityAnalyzer()
    # Caminho relativo ao arquivo CSV
    caminho_csv = os.path.join(os.path.dirname(__file__), 'customwords.csv')
    custom_words = carregar_custom_words(caminho_csv)
    analyzer.lexicon.update(custom_words)
    scores = analyzer.polarity_scores(texto)
    compound = scores['compound']
    if compound >= 0.05:
        return "Positivo"
    elif compound <= -0.05:
        return "Negativo"
    #else:
        #return "Neutro"

if __name__ == "__main__":
    texto = "O fogo já queimou 88 milhões de hectares de Cerrado entre 1985 e 2023, uma média de 9,5 milhões de hectares todos os anos. Área queimada equivale a 43% de toda a extensão do bioma e supera o território de países como Chile e Turquia. Em média, o bioma perdeu 9,5 milhões de hectares por ano para as chamas, superando os índices da Amazônia, que queimou 7,1 milhões de hectares anualmente."
    resultado = analisar_com_vader(texto)
    print(f"Sentimento detectado: {resultado}")
