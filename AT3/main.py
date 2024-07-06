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

def get_class_interval(minimun, number_of_class, c):
    return [(minimun + i * c, minimun + (i + 1) * c) for i in range(number_of_class)]

def calculate_xis(intervals):
    return [(a + b) / 2.0 for a, b in intervals]

def calculate_freq(data, intervals):
    freqs = [0 for _ in intervals]
    for d in data:
        for i, interval in enumerate(intervals):
            if interval[0] <= d < interval[1]:
                freqs[i] += 1
                break
    return freqs[:]

def calculate_pis(freqs, n):
    return [freq / n for freq in freqs]

def calculate_xi_dot_pi(xis, pis):
    return [xi * pi for xi, pi in zip(xis, pis)]

def calculate_idk(xis, pis, mean):
    return [(xi - mean)**2 * pi for xi, pi in zip(xis, pis)]

def calculate_zs(intervals, mean, std):
    return zip(*[((a - mean) / std, (b - mean) / std) for a, b in intervals])

def calculate_probability(z1s, z2s):
    probabilitys = [stats.norm.cdf(z2) - stats.norm.cdf(z1) for z1, z2 in zip(z1s, z2s)]
    # fazendo as extremidades irem de menos infinito ao primeiro numero do intervalo
    # e do ultimo numero do intervalo a mais infinito
    probabilitys[0] += stats.norm.cdf(z1s[0])
    probabilitys[-1] = stats.norm.cdf(float("+inf")) - stats.norm.cdf(z1s[-1])
    return probabilitys

def calculate_es(p_z1s_z2s, n):
    return [n * p_z1_z2 for p_z1_z2 in p_z1s_z2s]

def calculate_qui_squares(freqs, es):
    return [(freq - e)**2 / e for freq, e in zip(freqs, es)]

def adjust(intervals, freqs, es, error=0.05):
    adjust_intervals = []
    adjust_freqs = []
    adjust_es = []

    i = 0
    acc_intervals = intervals[0]
    acc_freqs = 0
    acc_es = 0
    while True:
        acc_es += es[i]
        acc_freqs += freqs[i]
        acc_intervals = (acc_intervals[0], intervals[i][1])
        i += 1

        if acc_es >= error:
            break

    adjust_intervals.append(acc_intervals)
    adjust_freqs.append(acc_freqs)
    adjust_es.append(acc_es)

    j = len(intervals) - 1
    acc_intervals = intervals[-1]
    acc_freqs = 0
    acc_es = 0
    while True:
        acc_es += es[j]
        acc_freqs += freqs[j]
        acc_intervals = (intervals[j][0], acc_intervals[1])
        j -= 1

        if acc_es >= error:
            break

    for h in range(i, j + 1):
        adjust_intervals.append(intervals[h])
        adjust_freqs.append(freqs[h])
        adjust_es.append(es[h])

    adjust_intervals.append(acc_intervals)
    adjust_freqs.append(acc_freqs)
    adjust_es.append(acc_es)

    return adjust_intervals, adjust_freqs, adjust_es


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

    # filter_1 = [lambda df: df['Marca'] == "ANTARCTICA PILSEN",
    #             lambda df: df["C03 - VIDRO 600ML RET"].notna(),
    #             lambda df: df["Seg"] == 2,
    #             lambda df: df["Produto"] == "CERVEJA"]

    amstel_lager = filter_excel(dataframe, filter_1)

    print("Tabela só com os dados do Amstel Lager")
    print(amstel_lager)
    print()
    
    print("Calculando o número de classes para Amstel Lager")
    print()

    n = len(amstel_lager)
    sturges = sturges_formula(n)
    root = square_root(n)
    number_of_class = sturges

    print(f"Número de dados {n}")
    print(f"Número de classes Sturges {sturges}")
    print(f"Número de classes Raiz {root}")
    print("Foi escolhido sturges")
    print()

    col = "C03 - VIDRO 600ML RET"
    prices = amstel_lager[col]

    minimum = prices.min()
    maximum = prices.max()
    r = maximum - minimum
    c =  round(r / number_of_class, 2)
    mean = prices.mean()
    median = prices.median()
    variance_value = prices.var()
    std_deviation_value = prices.std()

    class_intervals = get_class_interval(minimum, round(number_of_class), c)
    xis = calculate_xis(class_intervals)

    freqs = calculate_freq(prices, class_intervals)
    
    pis = calculate_pis(freqs, n)
    xis_dot_pis = calculate_xi_dot_pi(xis, pis)
    idk = calculate_idk(xis, pis, mean)

    z1s, z2s = calculate_zs(class_intervals, mean, std_deviation_value)
    p_z1s_z2s = calculate_probability(z1s, z2s)
    es = calculate_es(p_z1s_z2s, n)
    adjust_intervals, adjust_freqs, adjust_es = adjust(class_intervals, freqs, es)
    qui_squares = calculate_qui_squares(adjust_freqs, adjust_es)

    print("MODELO DESCRITIVO")
    print(f"Valor mínimo {minimum}")
    print(f"Valor máximo {maximum}")
    print(f"Valor de R {r}")
    print(f"Valor de C {c}")
    print(f"Média {mean}")
    print(f"Mediana {median}")
    print(f"Variância {variance_value}")
    print(f"Desvio Padrão {std_deviation_value}")
    print(f"Intervalos da classes {class_intervals}")
    print(f"Ponto médio dos intervalos {xis}")
    print(f"Frequências dos dados {freqs}")
    print(f"Soma das frequências {sum(freqs)}")
    print(f"Probabilidade de cada intervalo {pis}")
    print(f"Probabilidade x Ponto médio de cada intervalo {xis_dot_pis}")
    print(f"Não faço ideia do que é {idk}")
    print()

    print("TESTE DE ADERÊNCIA")
    print(f"Valores de z1 {z1s}")
    print(f"Valores de z2 {z2s}")
    print(f"P(z1 < Z < z2) {p_z1s_z2s}")
    print(f"Soma de probabilidade {sum(p_z1s_z2s)}")
    print(f"Probabilidades esperadas {es}")
    print(f"Soma das probabilidades esperadas {sum(es)}")
    print()

    print("AJUSTE TESTE DE ADERÊNCIA")
    print(f"Intervalo ajustado {adjust_intervals}")
    print(f"Frequência ajustada {adjust_freqs}")
    print(f"Probabilidade esperada ajustada {adjust_es}")
    print(f"Qui squares {qui_squares}")
    print(f"Qui square {sum(qui_squares)}")
    print()

    # print("Criando a tabela")
    # print()

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

    chi2, p = stats.chisquare(adjust_freqs, f_exp=es)
    print(chi2)
    print(p)

    # import numpy as np
    # import scipy.stats as stats

    # # Suponha que você tenha um conjunto de dados chamado data
    # data = np.random.lognormal(mean=0, sigma=1, size=100)  # Exemplo de dados log-normal

    # # Aplique a transformação logarítmica aos dados
    # log_data = np.log(data)

    # # Teste de Shapiro-Wilk para verificar a normalidade dos dados transformados
    # shapiro_test = stats.shapiro(log_data)
    # print(f"Shapiro-Wilk Test: Estatística = {shapiro_test.statistic}, p-valor = {shapiro_test.pvalue}")

    # # Teste de Kolmogorov-Smirnov para a distribuição log-normal
    # mean, sigma = np.mean(log_data), np.std(log_data, ddof=1)
    # ks_test = stats.kstest(data, 'lognorm', args=(sigma, 0, np.exp(mean)))
    # print(f"Kolmogorov-Smirnov Test: Estatística = {ks_test.statistic}, p-valor = {ks_test.pvalue}")



if __name__ == "__main__":
    main()
