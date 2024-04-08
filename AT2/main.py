import pandas as pd
import math
import matplotlib.pyplot as plt


def make_histogram(df):
    N_df = len(df)
    min_df = df["C03 - VIDRO 600ML RET"].min()
    max_df = df["C03 - VIDRO 600ML RET"].max()
    R_df = df["C03 - VIDRO 600ML RET"].max() - df["C03 - VIDRO 600ML RET"].min()
    raiz_N_df = math.sqrt(N_df)
    sturges_N_df = 1 + 3.32 * math.log10(N_df)

    C_raiz_N = R_df / raiz_N_df
    C_sturges_N = R_df / sturges_N_df

    C_df, num_class = ((C_raiz_N, raiz_N_df) if C_raiz_N > C_sturges_N else (C_sturges_N, sturges_N_df))
    C_df = round(C_df + C_df * 0.1, 2)
    num_class = int(round(num_class, 0))

    ranges_df = dict([((round(min_df + C_df * i, 2), round(min_df + C_df * (i + 1), 2)), 0) for i in range(num_class)])

    for value in df["C03 - VIDRO 600ML RET"].values:
        for k in ranges_df.keys():
            if k[0] <= value < k[1]:
                ranges_df[k] += 1
                break

    data = {
        "k": [],
        "CLASSES": [],
        "Freq": [],
        "pi": [],
        "Pi": [],
        "Xi": [],
        "Xi*pi": [],
        "di^2*pi": [],
    }

    pi_acc = 0
    for i, k in enumerate(ranges_df.keys()):
        data["k"].append(i + 1)
        data["CLASSES"].append(k)
        data["Freq"].append(ranges_df[k])
        pi = ranges_df[k]/N_df
        pi_acc += pi
        data["pi"].append(round(pi, 2))
        data["Pi"].append(pi_acc)
        Xi = (k[0] + k[1]) / 2
        data["Xi"].append(Xi)
        data["Xi*pi"].append(pi * Xi)

    media_ponderada = sum(data["Xi*pi"])
    media_simples = df["C03 - VIDRO 600ML RET"].sum() / N_df

    for i, k in enumerate(ranges_df.keys()):
        data["di^2*pi"].append((data["Xi"][i] - media_ponderada)**2 * data["pi"][i])
    
    var = sum(data["di^2*pi"])
    desvpad = math.sqrt(var)

    erro_relativo_grupo = 100.0 * abs(media_simples - media_ponderada) / media_simples
    cv = desvpad / media_ponderada

    ranges_df = pd.DataFrame(data)

    moda = ranges_df['Xi'][ranges_df['Freq'].idxmax()]
    assimetria = (media_ponderada - moda) / desvpad
    
    print(ranges_df)

    spec_df = pd.DataFrame({
        "N": [N_df],
        "min": [min_df],
        "max": [max_df],
        "raiz(n)": [raiz_N_df],
        "sturges(n)": [sturges_N_df],
        "class": [num_class],
        "C raiz(n)": [C_raiz_N],
        "C sturges(n)": [C_sturges_N],
        "C": [C_df],
    })

    param_df = pd.DataFrame({
        "TOTAL": N_df,
        "media ponderada": [media_ponderada],
        "media simples": [media_simples],
        "var": [var],
        "desvpad": [desvpad],
        "erro relativo grupo": [erro_relativo_grupo],
        "CV": [cv],
        "moda": [moda],
        "assimetria": [assimetria],
    })

    print(param_df)

    print(pd.concat([spec_df, ranges_df, param_df], axis=1))

    plt.hist(df["C03 - VIDRO 600ML RET"], bins=num_class, edgecolor='black')
    plt.xlabel('Preço ($)')
    plt.ylabel('Frequência')
    plt.title('Histograma dos Preços das Bebidas')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.boxplot(df["C03 - VIDRO 600ML RET"])
    plt.title('Boxplot dos Preços das Bebidas')
    plt.ylabel('Preço ($)')
    plt.xticks([1], ['Bebidas'])
    plt.grid(True)
    plt.show()



file_path = './doc/BASE DADOS_DESAFIO INDIVIDUAL.xlsx'
df = pd.read_excel(file_path)

amstel_lager = df[(df['Marca'] == "AMSTEL LAGER") &
                  (df["C03 - VIDRO 600ML RET"].notna()) &
                  (df["Seg"] == 2) &
                  (df["Produto"] == "CERVEJA")]

antarctica_pilsen = df[(df['Marca'] == "ANTARCTICA PILSEN") &
                       (df["C03 - VIDRO 600ML RET"].notna()) &
                       (df["Seg"] == 2) &
                       (df["Produto"] == "CERVEJA")]

make_histogram(amstel_lager)
make_histogram(antarctica_pilsen)