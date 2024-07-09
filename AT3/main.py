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

def calculate_qdp(xis, pis, mean):
    return [(xi - mean)**2 * pi for xi, pi in zip(xis, pis)]

def detect_outliers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return outliers, lower_bound, upper_bound

def calculate_descriptive_model(data):
    n = len(data)
    sturges = sturges_formula(n)
    root = square_root(n)
    number_of_class = sturges

    minimum = data.min()
    maximum = data.max()
    r = maximum - minimum
    c =  round(r / number_of_class, 4)
    mean = data.mean()
    median = data.median()
    variance_value = data.var()
    std_deviation_value = data.std()
    moda = data.mode()[0]

    class_intervals = get_class_interval(minimum, math.ceil(number_of_class), c)
    xis = calculate_xis(class_intervals)

    freqs = calculate_freq(data, class_intervals)
    
    pis = calculate_pis(freqs, n)
    Pis = [sum(pis[:i + 1]) for i in range(len(pis))]
    xis_dot_pis = calculate_xi_dot_pi(xis, pis)
    weighted_average = sum(xis_dot_pis)
    qdp = calculate_qdp(xis, pis, mean)
    outliers, lower_bound, upper_bound = detect_outliers(data)

    relative_error = abs(mean - weighted_average) / mean
    CV = std_deviation_value / weighted_average
    assi = (weighted_average - moda) / std_deviation_value
    dist = data.skew()

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
    print(f"Moda {moda}")
    print()
    print(f"Intervalos da classes {class_intervals}")
    print(f"Ponto médio dos intervalos {xis}")
    print(f"Frequências dos dados {freqs}")
    print(f"Soma das frequências {sum(freqs)}")
    print(f"Probabilidade de cada intervalo {pis}")
    print(f"Probabilidade de cada intervalo acumulada {Pis}")
    print(f"Probabilidade x Ponto médio de cada intervalo {xis_dot_pis}")
    print(f"Média ponderada {weighted_average}")
    print(f"QDP {qdp}")
    print(f"Variância sum(qdp) {sum(qdp)}")
    print(f"Desvio padrão sqrt(sum(qdp)) {math.sqrt(sum(qdp))}")
    print()
    print(f"Erro relativo {relative_error}")
    print(f"CV {CV}")
    print(f"Assimetria {assi}")
    print(f"Distorção {dist}")
    print(f"Limite inferior {lower_bound} e limite superior {upper_bound}")
    print(f"Outliers detectados: {outliers.tolist()}")
    print("="*238)
    print()

    return {
        "dados": data,
        "numero de dados": n,
        "numero de classes": math.ceil(number_of_class),
        "max": maximum,
        "min": minimum,
        "range": r,
        "c": c,
        "media": mean,
        "mediana": median,
        "var": variance_value,
        "desvpad": math.sqrt(sum(qdp)),
        "intervalos": class_intervals,
        "freqs": freqs,
        "Xis": xis,
        "pis": pis,
        "Pis": Pis,
        "xis*pis": xis_dot_pis,
        "media ponderada": weighted_average,
        "QDP": qdp,
        "erro relativo": relative_error,
        "CV": CV,
        "assimetria": assi,
        "distorcao": dist,
        "limite inferior": lower_bound,
        "limite superior": upper_bound,
        "outliers": outliers.tolist()
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
    acc_freqs = freqs[i]
    acc_es = es[i]
    i += 1
    while True:
        if acc_es >= error:
            break
        acc_es += es[i]
        acc_freqs += freqs[i]
        acc_intervals = (acc_intervals[0], intervals[i][1])
        i += 1


    adjust_intervals.append(acc_intervals)
    adjust_freqs.append(acc_freqs)
    adjust_es.append(acc_es)

    j = len(intervals) - 1
    acc_intervals = intervals[-1]
    acc_freqs = freqs[j]
    acc_es = es[j]
    j -= 1
    while True:
        if acc_es >= error:
            break
        acc_es += es[j]
        acc_freqs += freqs[j]
        acc_intervals = (intervals[j][0], acc_intervals[1])
        j -= 1


    for h in range(i, j + 1):
        adjust_intervals.append(intervals[h])
        adjust_freqs.append(freqs[h])
        adjust_es.append(es[h])

    adjust_intervals.append(acc_intervals)
    adjust_freqs.append(acc_freqs)
    adjust_es.append(acc_es)

    return adjust_intervals, adjust_freqs, adjust_es

def chi_square(n, class_intervals, mean, std_deviation_value, freqs):

    z1s, z2s = calculate_zs(class_intervals, mean, std_deviation_value)
    p_z1s_z2s = calculate_probability(z1s, z2s)
    es = calculate_es(p_z1s_z2s, n)
    adjust_intervals, adjust_freqs, adjust_es = adjust(class_intervals, freqs, es)
    qui_squares = calculate_qui_squares(adjust_freqs, adjust_es)
    df = len(adjust_intervals) - 2 - 1
    chi2_critical = stats.chi2.ppf(0.95, df)

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
    print(f"Soma da frequência ajustada {sum(adjust_freqs)}")
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

def graphs(data, class_number, path=None):
    # Criação da figura e subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Histograma
    axs[0].hist(data, bins=class_number, alpha=0.7, rwidth=0.85)
    axs[0].set_title('Histograma dos Dados')
    # axs[0].set_xlabel('Valor')
    # axs[0].set_ylabel('Frequência')

    # Q-Q Plot
    stats.probplot(data, dist="norm", plot=axs[1])
    axs[1].set_title('Q-Q Plot dos Dados')

    # Boxplot
    axs[2].boxplot(data, vert=True)
    axs[2].set_title('Boxplot dos Dados')
    axs[2].set_xticks([])  # Remove os ticks do eixo y

    # Ajuste dos layouts
    plt.tight_layout()

    if path:
        plt.savefig(path)

    plt.show()


def part_2(data, name):
    model = calculate_descriptive_model(data)
    n = model["numero de dados"]
    number_of_class = model["numero de classes"]
    class_intervals = model["intervalos"]
    freqs = model["freqs"]
    weighted_average = model["media ponderada"]
    std_deviation_value = model["desvpad"]

    chi_square_result = chi_square(n, class_intervals, weighted_average, std_deviation_value, freqs)
    graphs(data, number_of_class, f"datas/{name}_normal.png")
    dict2xlsx(model, f"datas/modelo-descritivo-{name}.xlsx")
    dict2xlsx(chi_square_result, f"datas/teste-aderencia-{name}.xlsx")

    print("\nLOG NORMAL\n")
    data = np.log(data)
    
    model = calculate_descriptive_model(data)
    n = model["numero de dados"]
    number_of_class = model["numero de classes"]
    class_intervals = model["intervalos"]
    freqs = model["freqs"]
    weighted_average = model["media ponderada"]
    std_deviation_value = model["desvpad"]

    chi_square_result = chi_square(n, class_intervals, weighted_average, std_deviation_value, freqs)
    graphs(data, number_of_class, f"datas/{name}_lognormal.png")

def part_3(sample1, sample2, path=None):
    # Estatísticas descritivas
    n1, n2 = len(sample1), len(sample2)
    mean1, mean2 = np.mean(sample1), np.mean(sample2)
    var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
    std1, std2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)

    # Teste para a razão entre variâncias
    f_statistic = var1 / var2
    dfn, dfd = n1 - 1, n2 - 1
    alpha = 0.05
    f_critical1 = stats.f.ppf(alpha / 2, dfn, dfd)
    f_critical2 = stats.f.ppf(1 - alpha / 2, dfn, dfd)
    reject_var_h0 = f_statistic < f_critical1 or f_statistic > f_critical2

    # Teste para a diferença de médias (assumindo variâncias iguais)
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    t_statistic = (mean1 - mean2) / np.sqrt(pooled_var * (1 / n1 + 1 / n2))
    df = n1 + n2 - 2
    t_critical = stats.t.ppf(1 - alpha / 2, df)
    reject_mean_h0 = abs(t_statistic) > t_critical

    # Construção do intervalo de confiança para a razão entre variâncias
    ci_var_lower = f_statistic / f_critical2
    ci_var_upper = f_statistic * f_critical2

    # Construção do intervalo de confiança para a diferença de médias
    ci_mean_lower = (mean1 - mean2) - t_critical * np.sqrt(pooled_var * (1 / n1 + 1 / n2))
    ci_mean_upper = (mean1 - mean2) + t_critical * np.sqrt(pooled_var * (1 / n1 + 1 / n2))

    # Resultados
    results = {
        'Estatística F': f_statistic,
        'Valor crítico F (lower)': f_critical1,
        'Valor crítico F (upper)': f_critical2,
        'Rejeição H0 variâncias iguais': reject_var_h0,
        'Estatística t': t_statistic,
        'Valor crítico t': t_critical,
        'Rejeição H0 médias iguais': reject_mean_h0,
        'IC razão variâncias (lower)': ci_var_lower,
        'IC razão variâncias (upper)': ci_var_upper,
        'IC diferença médias (lower)': ci_mean_lower,
        'IC diferença médias (upper)': ci_mean_upper
    }

    if path:
        dict2xlsx(results, "datas/estatisticas-parametricas.xlsx")

    return results

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

    embalagem = "C03 - VIDRO 600ML RET"
    produto = "CERVEJA"
    marca1 = "AMSTEL LAGER"
    marca2 = "ANTARCTICA PILSEN"
    seg1 = 2
    seg2 = 2

    filter_1 = [lambda df: df['Marca'] == marca1,
                lambda df: df[embalagem].notna(),
                lambda df: df["Seg"] == seg1,
                lambda df: df["Produto"] == produto]

    filter_2 = [lambda df: df['Marca'] == marca2,
                lambda df: df[embalagem].notna(),
                lambda df: df["Seg"] == seg2,
                lambda df: df["Produto"] == produto]

    data1 = filter_excel(dataframe, filter_1)
    data2 = filter_excel(dataframe, filter_2)

    data1_prices = data1[embalagem]
    data2_prices = data2[embalagem]

    part_2(data1_prices, marca1)
    part_2(data2_prices, marca2)

    part_3(data1_prices, data2_prices, f"datas/estatisticas-parametricas-{marca1}-{marca2}.xlsx")

    # https://ovictorviana.medium.com/teste-de-hip%C3%B3tese-com-python-ba5d751f156c

if __name__ == "__main__":
    main()
