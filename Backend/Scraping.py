import requests
from bs4 import BeautifulSoup

def extrair_texto_noticia(link):
    """
    Faz o scraping do corpo da matéria jornalística a partir do link e retorna o texto limpo.
    """
    try:
        response = requests.get(link, timeout=10)
        protocolo = response.url.split(':')[0]
        print(f"Protocolo utilizado: {protocolo.upper()}")
        response.raise_for_status()
    except Exception as e:
        print(f"Erro ao acessar o link: {e}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    artigo = soup.find('article')
    if artigo:
        paragrafos = artigo.find_all('p')
    else:
        paragrafos = soup.find_all('p')

    texto = ' '.join(p.get_text(strip=True) for p in paragrafos)
    texto = texto.replace('\n', ' ').replace('\r', ' ')
    texto = ' '.join(texto.split())  # Remove espaços duplicados
    texto = texto.replace('"', ' ')  # Remove todas as aspas duplas
    return texto if texto else None

def salvar_noticia_csv(texto, caminho_csv="Backend/BART/noticias5dias.csv"):
    """
    Salva o texto no CSV, entre aspas duplas, seguido de uma linha em branco.
    """
    if texto:
        texto_final = f'"{texto}"'
        with open(caminho_csv, 'a', encoding='utf-8') as csvfile:
            csvfile.write(texto_final + '\n\n')
        print("Matéria adicionada ao CSV com sucesso.")
    else:
        print("Texto vazio. Nada foi salvo.")

def main():
    while True:
        link = input("Cole o link da matéria jornalística: ")
        texto = extrair_texto_noticia(link)
        if texto:
            salvar_noticia_csv(texto)
        else:
            print("Não foi possível extrair o corpo da matéria.")
        continuar = input("Deseja adicionar outra notícia? (s/n): ").strip().lower()
        if continuar == 'n':
            print("Encerrando o programa.")
            break

if __name__ == "__main__":
    main()
