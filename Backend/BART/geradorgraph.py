import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import Counter

# Defina o caminho do CSV
csv_path = 'Backend/BART/outputbart.csv'

# Lista de assuntos/palavras-chave
assuntos = ['desmatamento', 'queimada', 'enchente', 'alagamento', 'seca', 'inundação', 'fogo', 'incêndio', 'tempestade', 'chuva']

def identificar_assunto(texto):
    texto = texto.lower()
    contagem = Counter({assunto: texto.count(assunto) for assunto in assuntos})
    assunto_mais_comum, freq = contagem.most_common(1)[0]
    if freq > 0:
        return assunto_mais_comum
    else:
        return 'outros'

if not os.path.exists(csv_path):
    print(f"Arquivo {csv_path} não encontrado.")
    exit(1)

# Leitura do CSV
# Agora as colunas são: texto, data, polaridade, assunto
colunas = ['texto', 'data', 'polaridade', 'assunto']
df = pd.read_csv(csv_path, names=colunas, header=None, encoding='utf-8', sep=',')

# Remover espaços extras e padronizar polaridade e assunto
df['polaridade'] = df['polaridade'].astype(str).str.strip().str.lower()
df['data'] = df['data'].astype(str).str.strip()
df['assunto'] = df['assunto'].astype(str).str.strip().str.lower()

print('Primeiras linhas do DataFrame:')
print(df.head())
print('Valores únicos de polaridade:', df['polaridade'].unique())
print('Valores únicos de assunto:', df['assunto'].unique())


# Gráfico 1: Quantidade de notícias por data e polaridade
contagem_data = df.groupby(['data', 'polaridade']).size().unstack(fill_value=0)
contagem_data.plot(kind='bar', stacked=False)
plt.title('Quantidade de notícias por data e polaridade')
plt.xlabel('Data')
plt.ylabel('Quantidade de notícias')
plt.tight_layout()
plt.legend(title='Polaridade')
plt.show()

# Gráfico 2: Quantidade de notícias por assunto e polaridade (apenas positivo/negativo)


df2 = df[df['polaridade'].isin(['positivo', 'negativo'])]
contagem_assunto = df2.groupby(['assunto', 'polaridade']).size().unstack(fill_value=0)
print('Contagem por assunto e polaridade:')
print(contagem_assunto)
if contagem_assunto.shape[0] == 0 or contagem_assunto.shape[1] == 0:
    print('Não há dados numéricos para plotar no gráfico 2.')
else:
    contagem_assunto.plot(kind='bar', stacked=False)
    plt.title('Quantidade de notícias por assunto e polaridade')
    plt.xlabel('Assunto')
    plt.ylabel('Quantidade de notícias')
    plt.tight_layout()
    plt.legend(title='Polaridade')
    plt.show()
contagem_assunto = df2.groupby(['assunto', 'polaridade']).size().unstack(fill_value=0)
contagem_assunto = contagem_assunto.loc[assuntos + ['outros']]  # Ordena conforme lista
contagem_assunto.plot(kind='bar', stacked=False)
plt.title('Quantidade de notícias por assunto e polaridade')
plt.xlabel('Assunto')
plt.ylabel('Quantidade de notícias')
plt.tight_layout()
plt.legend(title='Polaridade')
plt.show()
