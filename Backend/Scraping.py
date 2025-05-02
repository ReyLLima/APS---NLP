import requests
from bs4 import BeautifulSoup
from typing import Optional, Dict, Any
from dataclasses import dataclass
from urllib.parse import urlparse
import logging

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ScrapingResult:
    """Classe para armazenar o resultado do scraping"""
    text: str
    title: Optional[str] = None
    url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    success: bool = True
    error_message: Optional[str] = None

class WebScraper:
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def _validate_url(self, url: str) -> bool:
        """Valida se a URL é bem formada"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extrai o título da página"""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text(strip=True)
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text(strip=True)
        return None

    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extrai metadados da página"""
        metadata = {}
        meta_tags = soup.find_all('meta')
        for tag in meta_tags:
            if tag.get('name'):
                metadata[tag['name']] = tag.get('content', '')
            elif tag.get('property'):
                metadata[tag['property']] = tag.get('content', '')
        return metadata

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extrai o conteúdo principal da página"""
        # Lista de tags que podem conter o conteúdo principal
        main_content_tags = [
            soup.find('article'),
            soup.find('main'),
            soup.find(class_='content'),
            soup.find(class_='article-content'),
            soup.find(id='content'),
            soup.find(id='main-content')
        ]

        for content in main_content_tags:
            if content:
                return content.get_text(separator=' ', strip=True)

        # Se não encontrar nenhum container específico, usa o body
        return soup.body.get_text(separator=' ', strip=True)

    def scrape(self, url: str) -> ScrapingResult:
        """Realiza o scraping da página"""
        if not self._validate_url(url):
            return ScrapingResult(
                text="",
                success=False,
                error_message="URL inválida"
            )

        try:
            logger.info(f"Iniciando scraping da URL: {url}")
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            title = self._extract_title(soup)
            metadata = self._extract_metadata(soup)
            main_text = self._extract_main_content(soup)

            # Limita o texto para evitar excesso de dados
            main_text = main_text[:3000]

            logger.info(f"Scraping concluído com sucesso para: {url}")
            return ScrapingResult(
                text=main_text,
                title=title,
                url=url,
                metadata=metadata
            )

        except requests.Timeout:
            error_msg = f"Timeout ao acessar {url}"
            logger.error(error_msg)
            return ScrapingResult(text="", success=False, error_message=error_msg)
            
        except requests.RequestException as e:
            error_msg = f"Erro ao acessar {url}: {str(e)}"
            logger.error(error_msg)
            return ScrapingResult(text="", success=False, error_message=error_msg)
            
        except Exception as e:
            error_msg = f"Erro inesperado ao processar {url}: {str(e)}"
            logger.error(error_msg)
            return ScrapingResult(text="", success=False, error_message=error_msg)

def main(url: str) -> Dict[str, Any]:
    """Função principal para ser chamada externamente"""
    scraper = WebScraper()
    result = scraper.scrape(url)
    
    return {
        'success': result.success,
        'text': result.text if result.success else '',
        'title': result.title,
        'error': result.error_message,
        'metadata': result.metadata
    }

if __name__ == "__main__":
    url = input("Cole o link da notícia: ")
    result = main(url)
    
    if result['success']:
        print("\nTítulo:", result['title'])
        print("\nTexto extraído:\n")
        print(result['text'][:500], "...")
    else:
        print("Erro:", result['error'])