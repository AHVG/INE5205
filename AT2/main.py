import pandas as pd
import math

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

    ranges_df = dict([((min_df + C_df * i, min_df + C_df * (i + 1)), 0) for i in range(num_class)])

    for value in df["C03 - VIDRO 600ML RET"].values:
        for k in ranges_df.keys():
            if k[0] <= value < k[1]:
                ranges_df[k] += 1
                break

    print(df.iloc[0, 7])
    print(" N: ", end="")
    print(N_df)

    print()
    print(" Min e max: ", end="")
    print(min_df, end=" ")
    print(max_df)
    print(" R: ", end="")
    print(R_df)

    print()
    print(" Raiz(N): ", end="")
    print(raiz_N_df)
    print(" Sturges(N): ", end="")
    print(sturges_N_df)
    print(" Num class: ", end="")
    print(num_class)

    print()
    print(" C raiz N: ", end="")
    print(C_raiz_N)
    print(" C sturges N: ", end="")
    print(C_sturges_N)
    print(" C: ", end="")
    print(C_df)

    print()
    print(" Histogram: ")
    acc = 0

    for k in ranges_df.keys():
        print(f" {k[0]:5.2f} |-- {k[1]:5.2f} | " + "/" * ranges_df[k])
        acc += ranges_df[k]
    
    print(" TOTAL           |", acc)
    print()
    print()

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