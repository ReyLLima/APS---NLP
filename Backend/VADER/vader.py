import unicodedata
from leia import SentimentIntensityAnalyzer  # Usando o LeIA local (VADER PT-BR)

# Dicionário de boosters/dampeners em português
BOOSTERS_PTBR = {
    "muito": 0.293, "bastante": 0.293, "extremamente": 0.293, "altamente": 0.293, "super": 0.293, "totalmente": 0.293,
    "completamente": 0.293, "absolutamente": 0.293, "profundamente": 0.293, "enormemente": 0.293, "demais": 0.293,
    "incrivelmente": 0.293, "particularmente": 0.293, "especialmente": 0.293, "notavelmente": 0.293,
    "pouco": -0.293, "levemente": -0.293, "ligeiramente": -0.293, "minimamente": -0.293, "quase": -0.293,
    "um pouco": -0.293, "raramente": -0.293, "apenas": -0.293, "só": -0.293
}

def preprocessar_texto(texto):
    """
    Remove acentuação e converte para minúsculas para melhorar o match com o léxico.
    """
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII')
    texto = texto.lower()
    return texto

def aplicar_boosters(texto, scores, boosters=BOOSTERS_PTBR):
    """
    Ajusta o score de sentimento levando em conta boosters/dampeners no texto.
    """
    palavras = texto.split()
    for i, palavra in enumerate(palavras):
        if palavra in boosters:
            for j in range(i+1, min(i+4, len(palavras))):
                termo = palavras[j]
                if termo in scores:
                    scores[termo] += boosters[palavra]
    return scores

def analisar_com_vader(texto):
    """
    Analisa o sentimento do texto usando o VADER PT-BR (LeIA) com suporte a boosters.
    """
    analyzer = SentimentIntensityAnalyzer()
    texto = preprocessar_texto(texto)
    palavras = texto.split()
    # Gera um dicionário de scores para cada palavra do texto que está no lexicon
    scores_palavras = {p: analyzer.lexicon[p] for p in palavras if p in analyzer.lexicon}
    scores_palavras = aplicar_boosters(texto, scores_palavras)
    score_boosted = sum(scores_palavras.values()) if scores_palavras else 0
    scores = analyzer.polarity_scores(texto)
    compound = scores['compound'] + (score_boosted / (len(palavras) + 1))
    compound = max(min(compound, 1), -1)
    if compound >= 0.1:
        return "Positivo"
    elif compound <= -0.1:
        return "Negativo"
    else:
        return "Neutro"

if __name__ == "__main__":
    texto = (
        "Restauração ecológica e o Legado Verdes do CerradoSegundo o Ministério do Meio Ambiente, o Cerrado abriga cerca de 11,6 mil espécies de plantas nativas já catalogadas. Presente neste bioma, o Legado Verdes do Cerrado, Reserva Particular de Desenvolvimento Sustentável, de propriedade da CBA – Companhia Brasileira de Alumínio, já catalogou em seu território o equivalente a 12% de todas essas espécies de flora já mapeadas no Cerrado, sendo que deste total, 2% são de espécies endêmicas, isto é, que só podem ser encontradas em tal ecossistema, o que torna a área um valioso território para a conservação do bioma.Ciente da importância de sua flora, o Legado Verdes do Cerrado (LVC), por meio do seu Centro de Biodiversidade do Cerrado (CBC), alia a expertise das pesquisas científicas realizadas no território à produção inteligente de mais de 50 espécies nativas, sendo algumas raras ou ameaçadas de extinção, com o foco em restauração ecológica e paisagismo.Coordenada pela Reservas Votorantim, gestora da área e responsável pelos serviços de restauração realizados pelo Legado Verdes do Cerrado, a equipe de restauração ecológica do LVC tem expertise para atuar em todo o Brasil, e até o momento já restaurou mais de 120 hectares de áreas nativas. Somente em 2023, por intermédio da sua gestora, conseguiu acessar mercados em outros estados, como Minas Gerais e, produziu mais de 20 mil mudas voltadas a projetos de restauração ecológica para compensação ambiental, corroborando seu compromisso em recuperar e manter o Cerrado em pé.A restauração ecológica no Cerrado não apenas representa uma esperança para seu futuro, mas também se torna uma necessidade urgente diante dos desafios ambientais globais.Investir na reintegração de espécies nativas, recuperar áreas degradadas e fortalecer a conexão entre os fragmentos remanescentes deste bioma são medidas cruciais para proteger sua biodiversidade. Isso garante a saúde e o equilíbrio do meio ambiente, como conserva os serviços ecossistêmicos essenciais para as gerações futuras"
    )
    resultado = analisar_com_vader(texto)
    print(f"Sentimento detectado: {resultado}")
