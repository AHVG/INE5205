# https://www.geeksforgeeks.org/box-plot-in-python-using-matplotlib/
# https://medium.com/20-21/boxplot-com-python-usando-matplotlib-e-seaborn-be42cea47a6c
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html

import math
import argparse

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt


def sturges_formula(n):
    return 1 + 3.322 * math.log(n, 10)

def square_root(n):
    return math.sqrt(n)

def chi_square_test(data, num_classes):
    # Histogram (frequências observadas)
    observed_freq, bins = np.histogram(data, bins=num_classes)
    
    # Frequências esperadas usando a distribuição normal
    mean = np.mean(data)
    std = np.std(data)
    expected_freq = np.diff(stats.norm.cdf(bins, loc=mean, scale=std)) * len(data)
    
    # Estatística Qui-quadrado
    chi_square_stat = np.sum((observed_freq - expected_freq)**2 / expected_freq)
    
    # Graus de liberdade
    df = num_classes - 1 - 2  # num_classes - 1 - parâmetros estimados (média e std)
    
    # Valor crítico
    critical_value = stats.chi2.ppf(0.95, df)
    
    return chi_square_stat, critical_value, df

def chi_square_test_lognormal(data, num_classes):
    # Transformar os dados
    log_data = np.log(data)
    
    # Histogram (frequências observadas)
    observed_freq, bins = np.histogram(log_data, bins=num_classes)
    
    # Frequências esperadas usando a distribuição normal
    mean = np.mean(log_data)
    std = np.std(log_data)
    expected_freq = np.diff(stats.norm.cdf(bins, loc=mean, scale=std)) * len(log_data)
    
    # Estatística Qui-quadrado
    chi_square_stat = np.sum((observed_freq - expected_freq)**2 / expected_freq)
    
    # Graus de liberdade
    df = num_classes - 1 - 2  # num_classes - 1 - parâmetros estimados (média e std)
    
    # Valor crítico
    critical_value = stats.chi2.ppf(0.95, df)
    
    return chi_square_stat, critical_value, df


def main():

    parser = argparse.ArgumentParser(description="Lê um arquivo Excel e imprime na tela.")
    parser.add_argument("--excel_path", help="Caminho do arquivo Excel para ser lido.", default="./datas/BASE DADOS_DESAFIO INDIVIDUAL.xlsx")
    args = parser.parse_args()

    def filter_excel(df, filters):
        all_filter = pd.Series([True] * len(df), index=df.index)

        for f in filters:
            all_filter &= f(df)

        return df[all_filter]

    excel_path = args.excel_path
    dataframe = pd.read_excel(excel_path)

    filter_1 = [lambda df: df['Marca'] == "AMSTEL LAGER",
                lambda df: df["C03 - VIDRO 600ML RET"].notna(),
                lambda df: df["Seg"] == 2,
                lambda df: df["Produto"] == "CERVEJA"]

    amstel_lager = filter_excel(dataframe, filter_1)

    print("Tabela só com os dados do Amstel Lager")
    print(amstel_lager)
    print()
    
    print("Calculando o número de classes para Amstel Lager")
    print()

    n = len(amstel_lager)
    sturges = round(sturges_formula(n))
    root = round(square_root(n))
    number_of_class = sturges

    print(f"Número de dados {n}")
    print(f"Número de classes Sturges {sturges}")
    print(f"Número de classes Raiz {root}")
    print("Foi escolhido sturges")
    print()

    col = "C03 - VIDRO 600ML RET"
    prices = amstel_lager[col]
    
    print("Calculando Estatísticas Descritivas")

    minimum = prices.min()
    maximum = prices.max()
    mean = prices.mean()
    variance_value = prices.var()
    std_deviation_value = prices.std()

    print(f"Valor mínimo {minimum}")
    print(f"Valor máximo {maximum}")
    print(f"Média {mean}")
    print(f"Variância {variance_value}")
    print(f"Desvio Padrão {std_deviation_value}")
    print()

    # print("Histograma dos dados")

    # # Construindo os histogramas
    # # Colocar os dois histogramas juntos com ...subplots(1, 2, ...
    # fig, axs = plt.subplots(1, 1, figsize=(14, 7), sharey=True)

    # # Histograma para df1
    # axs.hist(prices, bins=number_of_class, edgecolor='black')
    # axs.set_title(f'Histograma de Preços ({n} Observações)')
    # axs.set_xlabel('Preços')
    # axs.set_ylabel('Frequência')

    # # plt.tight_layout()
    # plt.show()

    print("Calculando Normal")
    chi_square_stat, critical_value, df = chi_square_test(prices, number_of_class)

    print(f"Qui-quadrado {chi_square_stat}")
    print(f"Valor Crítico {critical_value}")
    print(f"Graus de Liberdade {df}")
    print()

    print("Calculando Log")
    chi_square_stat, critical_value, df = chi_square_test_lognormal(prices, number_of_class)

    print(f"Qui-quadrado {chi_square_stat}")
    print(f"Valor Crítico {critical_value}")
    print(f"Graus de Liberdade {df}")
    print()


if __name__ == "__main__":
    main()
