from textblob import TextBlob

def analisar_com_textblob(texto):
    blob = TextBlob(texto)
    polaridade = blob.sentiment.polarity
    if polaridade > 0:
        return "Positivo"
    elif polaridade < 0:
        return "Negativo"
    else:
        return "Neutro"

if __name__ == "__main__":
    texto = "Eu irei morrer"
    resultado = analisar_com_textblob(texto)
    print(f"Sentimento detectado: {resultado}")
