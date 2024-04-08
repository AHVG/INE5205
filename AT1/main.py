import pandas as pd

import matplotlib.pyplot as plt

# Caminho para o arquivo da base de dados (ajuste conforme necessário)
dados_amostra_path = 'BASE DADOS PARA A AMOSTRA.xlsx'

# Lendo os dados do arquivo, pulando as linhas de cabeçalho (ajuste o número conforme necessário)
dados_amostra = pd.read_excel(dados_amostra_path, skiprows=9)
print(dados_amostra)

# Filtrar os dados por tipo de combustível
dados_gasolina_comum = dados_amostra[dados_amostra['PRODUTO'] == 'GASOLINA COMUM']
dados_gasolina_aditivada = dados_amostra[dados_amostra['PRODUTO'] == 'GASOLINA ADITIVADA']

# Selecionando amostras aleatórias
amostra_gas_comum = dados_gasolina_comum.sample(n=10, random_state=1)
amostra_gas_aditivada = dados_gasolina_aditivada.sample(n=10, random_state=1)

# Função para calcular estatísticas
def calcular_estatisticas(dados, coluna_preco):
    return {
        'Média': dados[coluna_preco].mean(),
        'Mediana': dados[coluna_preco].median(),
        'Variância': dados[coluna_preco].var(),
        'Desvio Padrão': dados[coluna_preco].std(),
        'Coeficiente de Variação': dados[coluna_preco].std() / dados[coluna_preco].mean()
    }

# Supondo que a coluna com o preço médio é chamada 'Preço Médio' (ajuste conforme necessário)
estatisticas_gas_comum = calcular_estatisticas(amostra_gas_comum, 'PREÇO MÉDIO REVENDA')
estatisticas_gas_aditivada = calcular_estatisticas(amostra_gas_aditivada, 'PREÇO MÉDIO REVENDA')

with pd.ExcelWriter('Resultados_Estatisticos.xlsx') as writer:
    amostra_gas_comum.to_excel(writer, sheet_name='Gasolina Comum')
    amostra_gas_aditivada.to_excel(writer, sheet_name='Gasolina Aditivada')
    pd.DataFrame([estatisticas_gas_comum]).to_excel(writer, sheet_name='Estatísticas Gasolina Comum', index=False)
    pd.DataFrame([estatisticas_gas_aditivada]).to_excel(writer, sheet_name='Estatísticas Gasolina Aditivada', index=False)

# Imprime as estatísticas calculadas
print(estatisticas_gas_comum)
print(estatisticas_gas_aditivada)

print(amostra_gas_comum)
print(amostra_gas_aditivada)


# Exemplo de código para gerar um gráfico de pontos
plt.scatter(amostra_gas_comum['MUNICÍPIO'], amostra_gas_comum['PREÇO MÉDIO REVENDA'], label='Gasolina Comum')
plt.scatter(amostra_gas_aditivada['MUNICÍPIO'], amostra_gas_aditivada['PREÇO MÉDIO REVENDA'], label='Gasolina Aditivada')
plt.legend()
plt.xlabel('MUNICÍPIO')
plt.ylabel('PREÇO')
plt.title('Comparação dos Preços de Gasolina Comum e Aditivada')
plt.savefig('comparacao_precos_gasolina.png')
plt.show()