import math
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from ydata_profiling import ProfileReport


class Grouping:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def midpoint(self):
        return (self.lower + self.upper) / 2.0

    def __contains__(self, item):
        return self.lower <= item < self.upper

    def __repr__(self):
        return f"{self.lower:5.2f} |- {self.upper:5.2f}"

class BaseStatsCalculator(ABC):
    def __init__(self, df, column_name):
        self.df = df
        self.column_name = column_name

        self.N = len(df)
        self.min = df[column_name].min()
        self.max = df[column_name].max()
        self.R = self.max - self.min

        self.calculate_num_class()
        self.C = round((self.R / self.num_class) * 1.1, 2)

        self.spec = pd.DataFrame({
            "N": [self.N],
            "min": [self.min],
            "max": [self.max],
            "R": [self.R],
            "num classes": [self.num_class],
            "C": [self.C],
        })

        self.model = {
            "k": [],
            "CLASSES": [],
            "Freq": [],
            "pi": [],
            "Pi": [],
            "Xi": [],
            "Xi*pi": [],
            "di^2*pi": [],
        }

        Pi = 0

        for i in range(self.num_class):
            k = i + 1
            classe = Grouping(self.min + self.C * i, self.min + self.C * (i + 1))
            freq = len([value for value in self.df[self.column_name] if value in classe])
            pi = freq / self.N
            Pi += pi
            Xi = classe.midpoint()
            Xi_pi = pi * Xi
            di_2_pi = 0

            self.model["k"].append(k)
            self.model["CLASSES"].append(classe)
            self.model["Freq"].append(freq)
            self.model["pi"].append(pi)
            self.model["Pi"].append(Pi)
            self.model["Xi"].append(Xi)
            self.model["Xi*pi"].append(Xi_pi)
        
        self.weighted_average = sum(self.model["Xi*pi"])
        self.simple_average = self.df[self.column_name].mean()

        for i in range(self.num_class):
            di_2_pi = (self.model["Xi"][i] - self.weighted_average)**2 * self.model["pi"][i]
            self.model["di^2*pi"].append(di_2_pi)

        self.model = pd.DataFrame(self.model)

        self.var = sum(self.model["di^2*pi"])
        self.desvpad = math.sqrt(self.var)
        self.erro_relativo_agrupamento = 100.0 * abs(self.simple_average - self.weighted_average) / self.simple_average
        self.cv = self.desvpad / self.weighted_average
        self.moda = self.model['Xi'][self.model['Freq'].idxmax()]
        self.assimetria = (self.weighted_average - self.moda) / self.desvpad

        self.descriptive_measures = pd.DataFrame({
            "Media simples": [self.simple_average],
            "Media ponderada": [self.weighted_average],
            "var": [self.var],
            "desvpad": [self.desvpad],
            "erro relativo agrupamento": [self.erro_relativo_agrupamento],
            "cv": [self.cv],
            "moda": [self.moda],
            "assimetria": [self.assimetria],
        })

    def __repr__(self) -> str:
        return str(pd.concat([self.spec, self.model, self.descriptive_measures], axis=1))

    @abstractmethod
    def calculate_num_class(self):
        pass

class RaizNCalculator(BaseStatsCalculator):
    def calculate_num_class(self):
        self.num_class = math.ceil(math.sqrt(self.N))

class SturgesCalculator(BaseStatsCalculator):
    def calculate_num_class(self):
        self.num_class = math.ceil(1 + 3.32 * math.log10(self.N))


def make_histogram(dataframes):
    plt.figure(figsize=(10, 6))

    for df in dataframes:
        plt.hist(df["C03 - VIDRO 600ML RET"], bins='sturges', alpha=0.5, label=df["Marca"].iloc[0])

    plt.legend()
    plt.title('Histogramas Comparativos')
    plt.xlabel('Valor')
    plt.ylabel('Frequência')

    plt.show()


def make_boxplot(dataframes):
    plt.figure(figsize=(8, 6))

    boxplot_elements = plt.boxplot([dataframe["C03 - VIDRO 600ML RET"] for dataframe in dataframes],
                labels=[dataframe["Marca"].iloc[0] for dataframe in dataframes])

    for i, box in enumerate(boxplot_elements['boxes']):
        y_min = boxplot_elements['caps'][2*i].get_ydata()[0]
        y_max = boxplot_elements['caps'][2*i + 1].get_ydata()[0]
        y_q1 = box.get_ydata()[1]
        y_q3 = box.get_ydata()[2]
        y_med = boxplot_elements['medians'][i].get_ydata()[1]
        x_pos = box.get_xdata()[0]
        
        plt.text(x_pos, y_min, f'{y_min:.2f}', verticalalignment='top', fontsize=8, color='red')
        plt.text(x_pos, y_max, f'{y_max:.2f}', verticalalignment='bottom', fontsize=8, color='red')
        plt.text(x_pos, y_q1, f'{y_q1:.2f}', verticalalignment='top', fontsize=8, color='red')
        plt.text(x_pos, y_q3, f'{y_q3:.2f}', verticalalignment='bottom', fontsize=8, color='red')
        plt.text(x_pos, y_med, f'{y_med:.2f}', verticalalignment='center', fontsize=8, color='red')

    plt.title('Análise exploratória - boxplot')
    plt.xlabel('Marca')
    plt.ylabel('Valor da cerveja (R$)')
    plt.show()

def main():

    parser = argparse.ArgumentParser(description="Lê um arquivo Excel e imprime na tela.")
    parser.add_argument("--excel_path", help="Caminho do arquivo Excel para ser lido.", default="./doc/BASE DADOS_DESAFIO INDIVIDUAL.xlsx")
    args = parser.parse_args()

    def filter_excel(df, filters):
        all_filter = pd.Series([True] * len(df), index=df.index)

        for filter in filters:
            all_filter &= filter(df)

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

    print(SturgesCalculator(amstel_lager, "C03 - VIDRO 600ML RET"))
    # print(SturgesCalculator(antarctica_pilsen, "C03 - VIDRO 600ML RET"))

    # make_boxplot([amstel_lager, antarctica_pilsen])
    # make_histogram([amstel_lager, antarctica_pilsen])

    plt.show()

    profile = ProfileReport(amstel_lager, title="Pandas Profiling Report", config_file='profiler_settings.yml')
    profile.to_file("your_report.html")

main()
