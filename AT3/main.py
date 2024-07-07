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

def calculate_descriptive_model(data):
    n = len(data)
    sturges = sturges_formula(n)
    root = square_root(n)
    number_of_class = sturges

    minimum = data.min()
    maximum = data.max()
    r = maximum - minimum
    c =  round(r / number_of_class, 4) * 1.05
    mean = data.mean()
    median = data.median()
    variance_value = data.var()
    std_deviation_value = data.std()

    class_intervals = get_class_interval(minimum, round(number_of_class), c)
    xis = calculate_xis(class_intervals)

    freqs = calculate_freq(data, class_intervals)
    
    pis = calculate_pis(freqs, n)
    xis_dot_pis = calculate_xi_dot_pi(xis, pis)
    idk = calculate_idk(xis, pis, mean)

    print("MODELO DESCRITIVO " + "="*220)
    print()
    print(f"Número de dados {n}")
    print(f"Número de classes sturgues {sturges}")
    print(f"Número de classes raiz {root}")
    print(f"Número de classes escolhido {number_of_class}")
    print()
    print(f"Valor mínimo {minimum}")
    print(f"Valor máximo {maximum}")
    print(f"Valor de R {r}")
    print(f"Valor de C {c}")
    print()
    print(f"Média {mean}")
    print(f"Mediana {median}")
    print(f"Variância {variance_value}")
    print(f"Desvio Padrão {std_deviation_value}")
    print()
    print(f"Intervalos da classes {class_intervals}")
    print(f"Ponto médio dos intervalos {xis}")
    print(f"Frequências dos dados {freqs}")
    print(f"Soma das frequências {sum(freqs)}")
    print(f"Probabilidade de cada intervalo {pis}")
    print(f"Probabilidade x Ponto médio de cada intervalo {xis_dot_pis}")
    print(f"Não faço ideia do que é {idk}")
    print("="*238)
    print()

    return {
        "dados": data,
        "numero de dados": n,
        "numero de classes": number_of_class,
        "max": maximum,
        "min": minimum,
        "range": r,
        "c": c,
        "media": mean,
        "mediana": median,
        "variancia": variance_value,
        "desvio padrao": std_deviation_value,
        "intervalos": class_intervals,
        "freqs": freqs,
        "xis": xis,
        "pis": pis,
        "xis dot pis": xis_dot_pis,
        "idk": idk
    }

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

def adjust(intervals, freqs, es, error=5.0):
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

def chi_square(n, number_of_class, class_intervals, mean, std_deviation_value, freqs):
    df = number_of_class - 2 - 1
    chi2_critical = stats.chi2.ppf(0.95, df)

    z1s, z2s = calculate_zs(class_intervals, mean, std_deviation_value)
    p_z1s_z2s = calculate_probability(z1s, z2s)
    es = calculate_es(p_z1s_z2s, n)
    adjust_intervals, adjust_freqs, adjust_es = adjust(class_intervals, freqs, es)
    qui_squares = calculate_qui_squares(adjust_freqs, adjust_es)

    print("TESTE DE ADERÊNCIA " + "="*219)
    print()
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
    print(f"Qui square critico {chi2_critical}")
    print("="*237)
    print()

    return {
        "z1s": z1s,
        "z2s": z2s,
        "Oi": p_z1s_z2s,
        "Ei": es,
        "sum(Oi)": sum(p_z1s_z2s),
        "sum(Ei)": sum(es),
        "intervalo ajustado": adjust_intervals,
        "frequência ajustada": adjust_freqs,
        "probabilidade ajustada": adjust_es,
        "qui squares": qui_squares,
        "qui square": sum(qui_squares),
        "qui square critico": chi2_critical,
    }

def dict2xlsx(data_dict, name="output.xlsx"):
    # Preparação do DataFrame
    # Primeiro, transformar valores únicos em listas do mesmo tamanho
    max_length = max(len(v) if isinstance(v, list) else 1 for v in data_dict.values())
    for key, value in data_dict.items():
        if not isinstance(value, list) and not isinstance(value, tuple):
            data_dict[key] = [value] + [None] * (max_length-1)
        else:
            # Se a lista for menor que o comprimento máximo, completar com None
            data_dict[key] = list(data_dict[key]) + [None] * (max_length - len(data_dict[key]))

    # Conversão do dicionário para DataFrame
    df = pd.DataFrame(data_dict)

    # Salvar o DataFrame em um arquivo Excel
    df.to_excel(f'{name}', index=False)

def graphs(data):
    # Criação da figura e subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Histograma
    axs[0].hist(data, bins='auto', alpha=0.7, rwidth=0.85)
    axs[0].set_title('Histograma dos Dados Transformados')
    axs[0].set_xlabel('Valor')
    axs[0].set_ylabel('Frequência')

    # Q-Q Plot
    stats.probplot(data, dist="norm", plot=axs[1])
    axs[1].set_title('Q-Q Plot dos Dados Transformados')

    # Ajuste dos layouts
    plt.tight_layout()
    plt.show()

def part_2(data, name):
    model = calculate_descriptive_model(data)
    n = model["numero de dados"]
    number_of_class = model["numero de classes"]
    class_intervals = model["intervalos"]
    freqs = model["freqs"]
    mean = model["media"]
    std_deviation_value = model["desvio padrao"]

    chi_square_result = chi_square(n, number_of_class, class_intervals, mean, std_deviation_value, freqs)
    graphs(data)
    dict2xlsx(model, f"datas/modelo-descritivo-{name}.xlsx")
    dict2xlsx(chi_square_result, f"datas/teste-aderencia-{name}.xlsx")

    print("\nLOG NORMAL\n")
    data = np.log(data)
    
    model = calculate_descriptive_model(data)
    n = model["numero de dados"]
    number_of_class = model["numero de classes"]
    class_intervals = model["intervalos"]
    freqs = model["freqs"]
    mean = model["media"]
    std_deviation_value = model["desvio padrao"]

    chi_square_result = chi_square(n, number_of_class, class_intervals, mean, std_deviation_value, freqs)
    graphs(data)

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

    filter_2 = [lambda df: df['Marca'] == "ANTARCTICA PILSEN",
                lambda df: df["C03 - VIDRO 600ML RET"].notna(),
                lambda df: df["Seg"] == 2,
                lambda df: df["Produto"] == "CERVEJA"]

    amstel_lager = filter_excel(dataframe, filter_1)
    antarctica_pilsen = filter_excel(dataframe, filter_2)

    col = "C03 - VIDRO 600ML RET"

    amstel_lager_prices = amstel_lager[col]
    antarctica_pilsen_prices = antarctica_pilsen[col]

    part_2(amstel_lager_prices, "AMSTEL-LAGER")
    part_2(antarctica_pilsen_prices, "ANTARCTICA-PILSEN")

    # https://ovictorviana.medium.com/teste-de-hip%C3%B3tese-com-python-ba5d751f156c

if __name__ == "__main__":
    main()
